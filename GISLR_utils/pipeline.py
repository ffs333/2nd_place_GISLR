import os
import gc
import time

import torch
import wandb

import numpy as np

from .core import get_optimizer, get_scheduler, train_fn, valid_fn
from .utils import set_seed, load_data, get_scores
from .data import prepare_loaders, prepare_loader_full
from .models import get_model
from .losses import get_loss_func

from torch.optim import swa_utils


def train_loop(CFG, folds, fold, LOGGER):
    whole_time_start = time.time()
    if CFG.finetune_change_seed:
        CFG.seed *= 2
    set_seed(CFG.seed)

    LOGGER.info(f"========== Fold: {fold} training ==========")

    train_loader, valid_loader, valid_inds = prepare_loaders(CFG, folds, fold)
    data_dict = {'Y': np.load(CFG.base_path + 'gen_xyz/Y.npy')}

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================
    model = get_model(CFG)
    if CFG.finetune:
        fn_fol = CFG.base_path + 'FINETUNE_DIR/'
        fn_path = fn_fol + '/checkpoints/CKPT.pth'
        loaded_check = torch.load(fn_path, map_location=torch.device('cpu'))
        model.to(torch.device('cpu'))
        model.load_state_dict(dict([(n, p) for n, p in loaded_check['model'].items() if 'proj_head' not in n]),
                              strict=False)

        best_score = 0
        best_topk_score = 0
        print(f'Loaded from checkpoint')
        print(f'Best score set to: {best_score:.5f}')
    else:
        best_score = 0
        best_topk_score = 0
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG)

    if CFG.use_swa:
        swa_sched = swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=1e-4)
        swa_model = swa_utils.AveragedModel(model)
        swa_start = CFG.swa_start
    else:
        swa_start = 0
        swa_sched, swa_model = None, None

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0

    save_path = CFG.save_path + f"{CFG.model}_fold{fold}_best.pth"
    save_path_topk = CFG.save_path + f"{CFG.model}_fold{fold}_best_topk.pth"
    last_path = CFG.save_path + f"{CFG.model}_fold{fold}_last.pth"

    if CFG.finetune:
        torch.save({'model': model.state_dict(),
                    'results': 1,
                    }, last_path)

        torch.save({'model': model.state_dict(),
                    'results': 1,
                    }, save_path)

    not_improve = 0

    for epoch in range(1, CFG.epochs + 1):

        if CFG.early_stopping is not None:
            if not_improve > CFG.early_stopping:
                print(f'Not improve for {CFG.early_stopping} epochs')
                break

        if CFG.finetune:
            if epoch < CFG.finetune_epoch:
                continue

        start_time = time.time()
        print(f'Epoch {epoch}/{CFG.epochs} | Fold {fold}')

        # train
        avg_loss = train_fn(cfg=CFG, fold=fold, train_loader=train_loader, model=model, criterion=criterion,
                            optimizer=optimizer, scheduler=scheduler, device=CFG.device, epoch=epoch,
                            _global_step=_global_step, swa_start_=swa_start, swa_sched_=swa_sched, swa_model_=swa_model)

        _global_step += len(train_loader)
        # eval

        if CFG.use_swa and epoch > swa_start:
            avg_val_loss, predictions_ar = valid_fn(cfg=CFG, valid_loader=valid_loader, model=swa_model,
                                                    epoch=epoch, criterion=criterion, device=CFG.device)
        else:
            avg_val_loss, predictions_ar = valid_fn(cfg=CFG, valid_loader=valid_loader, model=model,
                                                    epoch=epoch, criterion=criterion, device=CFG.device)

        predictions, predictions_dsv = predictions_ar

        # scoring
        accuracy_base, topk_score_base = get_scores(data_dict['Y'][valid_inds], predictions, CFG)
        print(f'ACCURACY SCORE: {accuracy_base:.6f}')
        print(f'TOPK SCORE: {topk_score_base:.6f}')

        if CFG.deep_supervision:
            accuracy_dsv, topk_score_dsv = get_scores(data_dict['Y'][valid_inds], predictions_dsv, CFG)
            print(f'ACCURACY SCORE DSV: {accuracy_dsv:.6f}')
            print(f'TOPK SCORE DSV: {topk_score_dsv:.6f}')
        else:
            accuracy_dsv, topk_score_dsv = 0, 0

        if accuracy_base > accuracy_dsv:
            print(f'BASE BETTER')
        else:
            print(f'DSV BETTER')

        accuracy = accuracy_base  # max(accuracy_base, accuracy_dsv)
        topk_score = topk_score_base  # max(topk_score_base, topk_score_dsv)

        params = {'fold': fold, 'epoch': epoch, 'accuracy': accuracy, f'topk{CFG.k}': topk_score}

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch} - avg_train_loss: {avg_loss:.5f}  '
            f'avg_val_loss: {avg_val_loss:.5f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch} - Accuracy: {accuracy:.5f} | TopK{CFG.k}: {topk_score:.5f}')

        if accuracy > best_score:
            LOGGER.info(f'||||||||| Best Score Updated {best_score:0.5f} -->> {accuracy:0.5f} | Model Saved |||||||||')
            LOGGER.info(f'Best params: {params}')
            best_score = accuracy
            if CFG.use_swa and epoch > swa_start:
                torch.save({'model': swa_model.state_dict(),
                            'results': params,
                            }, save_path)
            else:
                torch.save({'model': model.state_dict(),
                            'results': params,
                            }, save_path)
            not_improve = 0
        else:
            LOGGER.info(f'Score NOT updated. Current best: {best_score:0.4f}')
            not_improve += 1
            if CFG.use_restart and epoch > CFG.restart_from:
                ld_ckpt = torch.load(save_path)
                model.load_state_dict(ld_ckpt['model'])
                print(f'Loaded previous best model with params: {ld_ckpt["results"]}')

        if topk_score > best_topk_score:
            LOGGER.info(f'|||| Best TOPK Score Updated {best_topk_score:0.5f} -->> {topk_score:0.5f} | Model Saved |||||')
            LOGGER.info(f'Best params: {params}')
            best_topk_score = topk_score
            if CFG.use_swa and epoch > swa_start:
                torch.save({'model': swa_model.state_dict(),
                            'results': params,
                            }, save_path_topk)
            else:
                torch.save({'model': model.state_dict(),
                            'results': params,
                            }, save_path_topk)
        else:
            LOGGER.info(f'Score NOT updated. Current best topk: {best_topk_score:0.4f}')

        if CFG.wandb:
            current_lr = optimizer.param_groups[0]['lr']
            try:
                if CFG.deep_supervision:
                    wandb.log({f"[fold{fold}] accuracy": accuracy_base,
                               f"[fold{fold}] TopK{CFG.k}": topk_score_base,
                               f"[fold{fold}] avg_train_loss": avg_loss,
                               f"[fold{fold}] avg_val_loss": avg_val_loss,
                               f"[fold{fold}] lr": current_lr,
                               f"[fold{fold}] accuracy DSV": accuracy_dsv,
                               f"[fold{fold}] TopK{CFG.k} DSV": topk_score_dsv})
                else:
                    wandb.log({f"[fold{fold}] accuracy": accuracy,
                               f"[fold{fold}] TopK{CFG.k}": topk_score,
                               f"[fold{fold}] avg_train_loss": avg_loss,
                               f"[fold{fold}] avg_val_loss": avg_val_loss,
                               f"[fold{fold}] lr": current_lr,
                               f"[fold{fold}] accuracy DSV": accuracy,
                               f"[fold{fold}] TopK{CFG.k} DSV": topk_score
                               })
            except:
                pass

    # Load and re save best valid
    loadeed_check = torch.load(save_path, map_location=torch.device('cpu'))
    cur_topk = round(loadeed_check['results'][f'topk{CFG.k}'], 4)
    final_v_path = CFG.save_path + f"{CFG.model}_fold{fold}_final_loop_best_main_{best_score:0.4f}_{cur_topk}.pth"
    torch.save(loadeed_check, final_v_path)

    loadeed_checktopk = torch.load(save_path_topk, map_location=torch.device('cpu'))
    cur_be = round(loadeed_checktopk['results']['accuracy'], 4)
    final_v_path_topk = CFG.save_path + f"{CFG.model}_fold{fold}_final_loop_best_topk_{cur_be}_{best_topk_score:0.4f}.pth"
    torch.save(loadeed_checktopk, final_v_path_topk)

    os.remove(save_path)
    os.remove(save_path_topk)

    # END OF TRAINING
    LOGGER.info(f'FOLD {fold} TRAINING FINISHED. BEST ACCURACY SCORE: {best_score:0.5f} | BEST TOPK SCORE: {best_topk_score:.4f}',
                f'SAVED HERE: {final_v_path}',
                f'AND HERE: {final_v_path_topk}')

    if CFG.use_swa:
        swa_model.to(torch.device('cpu'))
        swa_model.load_state_dict(loaded_check['model'])
        swa_model.to(CFG.device)

        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        final_s_path_swa = CFG.save_path + f"{CFG.model}_fold{fold}_final_loop_SWABN_{cur_be}_{best_topk_score:0.4f}.pth"
        torch.save({'model': swa_model.state_dict()}, final_s_path_swa)

    torch.cuda.empty_cache()
    gc.collect()

    time_elapsed = time.time() - whole_time_start
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    return best_score, best_topk_score


def full_train_loop(CFG, folds, LOGGER):
    whole_time_start = time.time()
    set_seed(CFG.seed)

    LOGGER.info(f"========== Full training ==========")

    print(f'Loading data.....')
    train_loader = prepare_loader_full(CFG, folds)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================
    model = get_model(CFG)
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0

    fold = 0
    for epoch in range(1, CFG.epochs + 1):

        start_time = time.time()
        print(f'Epoch {epoch}/{CFG.epochs}')

        # train
        avg_loss = train_fn(cfg=CFG, fold=fold, train_loader=train_loader, model=model, criterion=criterion,
                            optimizer=optimizer, scheduler=scheduler, device=CFG.device, epoch=epoch,
                            _global_step=_global_step, swa_start_=99999, swa_sched_=None, swa_model_=None)

        _global_step += len(train_loader)
        # eval

        if epoch > CFG.epochs - 15:
            last_path = CFG.save_path + f"{CFG.model}_FULL_ep{epoch}.pth"
            torch.save({'model': model.state_dict()}, last_path)

        if epoch % 10 == 0 and epoch > CFG.epochs - 55:
            last_path = CFG.save_path + f"{CFG.model}_FULL_ep{epoch}.pth"
            torch.save({'model': model.state_dict()}, last_path)

        elapsed = time.time() - start_time

        if CFG.wandb:
            try:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({f"[FULL] avg_train_loss": avg_loss,
                           f"[FULL] lr": current_lr})
            except:
                pass

        if epoch > CFG.BREAK_EPOCH:
            last_path = CFG.save_path + f"{CFG.model}_FULL_BREAKEPOCH{epoch}.pth"
            torch.save({'model': model.state_dict()}, last_path)
            print(f'Train finished on break epoch')
            break

    LOGGER.info(f'FOLD {fold} TRAINING FINISHED.')

    torch.cuda.empty_cache()
    gc.collect()

    time_elapsed = time.time() - whole_time_start
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

