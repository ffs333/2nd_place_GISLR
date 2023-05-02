import gc

from tqdm.auto import tqdm
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch_optimizer import Lookahead, RAdam, Yogi

from .utils import AverageMeter


def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas)
    elif cfg.optimizer == 'AdamW':
        if cfg.model == 'TREF':
            optimizer = optim.AdamW([{'params': model.eff.parameters(), 'lr': cfg.lr},
                                     {'params': model.transformer.parameters(), 'lr': 1e-4},
                                     {'params': model.logits.parameters(), 'lr': cfg.lr}],
                                     eps=cfg.eps, betas=cfg.betas,
                                     weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                                    weight_decay=cfg.weight_decay)

    elif cfg.optimizer == 'RAdam':
        if cfg.model == 'TREF':
            optimizer = RAdam([{'params': model.eff.parameters(), 'lr': cfg.lr},
                                     {'params': model.transformer.parameters(), 'lr': 1e-4},
                                     {'params': model.logits.parameters(), 'lr': cfg.lr}], eps=cfg.eps, betas=cfg.betas,
                              weight_decay=cfg.weight_decay)
        else:
            optimizer = RAdam(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                              weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Yogi':
        optimizer = Yogi(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                         weight_decay=cfg.weight_decay, initial_accumulator=1e-6)

    # Lookahed
    elif cfg.optimizer == 'Lookahead_AdamW':
        optimizer = Lookahead(optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                                          weight_decay=cfg.weight_decay), alpha=0.5, k=5)
    elif cfg.optimizer == 'Lookahead_RAdam':
        if cfg.model == 'TREF':
            optimizer = Lookahead(RAdam([{'params': model.eff.parameters(), 'lr': cfg.lr},
                                     {'params': model.transformer.parameters(), 'lr': 1e-4},
                                     {'params': model.logits.parameters(), 'lr': cfg.lr}],
                                        eps=cfg.eps, betas=cfg.betas,
                                        weight_decay=cfg.weight_decay), alpha=0.5, k=5)
        else:
            optimizer = Lookahead(RAdam(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                                    weight_decay=cfg.weight_decay), alpha=0.5, k=5)
    elif cfg.optimizer == 'Lookahead_Yogi':
        optimizer = Lookahead(Yogi(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                                   weight_decay=cfg.weight_decay, initial_accumulator=1e-6), alpha=0.5, k=5)
    else:
        raise ValueError('Error in "get_optimizer" function:',
                         f'Wrong optimizer name. Choose one from ["Adam", "AdamW"] ')

    return optimizer


def get_scheduler(cfg, scheduler_name, optimizer, num_train_steps, cycles):
    if scheduler_name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps)
    elif scheduler_name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps,
            num_training_steps=num_train_steps, num_cycles=cycles)

    elif scheduler_name == 'cosine_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=cfg.min_lr,
                                                                   T_0=int(num_train_steps // cycles), T_mult=1)

    elif scheduler_name == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                  total_steps=int(num_train_steps * cfg.onecycle_m),
                                                  pct_start=cfg.onecycle_start)

    elif scheduler_name == 'simple_cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_train_steps // cycles),
                                                               eta_min=cfg.min_lr, last_epoch=-1)

    elif scheduler_name == 'cosine_warmup_ext':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=num_train_steps // 2.5,
                                                  cycle_mult=1.75,
                                                  max_lr=cfg.lr,
                                                  min_lr=cfg.min_lr,
                                                  warmup_steps=cfg.num_warmup_steps,
                                                  gamma=0.7)


    else:
        raise ValueError('Error in "get_scheduler" function:',
                         f'Wrong scheduler name. Choose one from ["linear", "cosine", "cosine_restart", "onecycle" ]')

    return scheduler


def train_fn(cfg, fold, train_loader, model, criterion,
             optimizer, scheduler, device, epoch, _global_step, swa_start_, swa_sched_, swa_model_):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Ep.{epoch} Train ')

    for step, batch in pbar:
        _global_step += 1

        x, label = batch
        x = x.to(device)

        label = label.to(device).float()
        batch_size = x.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            if cfg.deep_supervision:
                y_pred, sv_pred = model(x)
            else:
                y_pred = model(x)

            if cfg.deep_supervision:
                sv_loss = None
                for sv in sv_pred:
                    if sv_loss is None:
                        sv_loss = criterion(sv, label)
                    else:
                        sv_loss += criterion(sv, label)

                sv_loss /= len(model.sup_inds)
                loss = criterion(y_pred, label) * 0.6 + 0.4 * sv_loss
            else:
                loss = criterion(y_pred, label)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()
        losses.update(loss.item(), batch_size)

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if cfg.use_swa and epoch > swa_start_:
                swa_model_.update_parameters(model)
                swa_sched_.step()
            else:
                scheduler.step()

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         lr=f'{current_lr:0.8f}',
                         gpu_mem=f'{mem:0.2f} GB',
                         global_step=f'{_global_step}')

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg


@torch.no_grad()
def valid_fn(cfg, valid_loader, model, epoch, criterion, device):
    losses = AverageMeter()
    model.eval()
    prediction = torch.tensor([], dtype=torch.float32)
    if cfg.deep_supervision:
        prediction_dsv = torch.tensor([], dtype=torch.float32)
    else:
        prediction_dsv = None

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Ep.{epoch} Eval ')

    for step, batch in pbar:

        x, label = batch
        x = x.to(device)

        label = label.to(device).float()
        batch_size = x.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            if cfg.deep_supervision:
                y_pred, sv_pred = model(x)
            else:
                y_pred = model(x)

            if cfg.deep_supervision:
                sv_loss = None
                for sv in sv_pred:
                    if sv_loss is None:
                        sv_loss = criterion(sv, label)
                    else:
                        sv_loss += criterion(sv, label)

                sv_loss /= len(model.sup_inds)
                loss = criterion(y_pred, label) * 0.6 + 0.4 * sv_loss
            else:
                loss = criterion(y_pred, label)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

        out = torch.softmax(y_pred, 1).detach().cpu()
        prediction = torch.cat([prediction, out])

        if cfg.deep_supervision:
            out_dsv = (0.6*out + 0.4*torch.sum(torch.stack([torch.softmax(x, 1).detach().cpu() for x in sv_pred]), dim=0)/len(sv_pred))
            prediction_dsv = torch.cat([prediction_dsv, out_dsv])

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, [prediction, prediction_dsv]
