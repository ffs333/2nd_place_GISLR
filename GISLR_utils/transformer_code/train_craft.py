import os
import json
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import top_k_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from ranger import Ranger

import pytorch_lightning as ptl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
import pl_bolts

from utils.bert import BertEncoderCPE
from utils.deberta import DebertaV2EncoderCPE
from utils.linear_warmup import LinearWarmupCosineAnnealingLR
from utils.csvlogger import LogJsonLogger
from utils.augmentations import *

with open("./data/sign_to_prediction_index_map.json") as f:
    sign2label = pl.DataFrame(list(zip(*json.load(f).items())), schema = ["sign", "label"])

df = (
    pl
    .read_csv("./data/train_with_length.csv")
    .join(sign2label, on = "sign", how = "left")
    .with_columns(("./data/" + pl.col("path")).str.replace("train_landmark_files", "train_npy").str.replace("parquet", "npy").alias("path"))
    .with_columns(pl.col("length").cumsum().alias("idx") - pl.col("length"))
    .to_pandas()
)
npy = np.load("./data/train_npy.npy")

class Data(Dataset):
    def __init__(self, df, cfg, phase):
        self.df = df
        self.max_len = cfg.config.max_position_embeddings - 1
        self.phase = phase
        self.dis_idx0, self.dis_idx1 = torch.where(torch.triu(torch.ones((21, 21)), 1) == 1)
        self.dis_idx2, self.dis_idx3 = torch.where(torch.triu(torch.ones((20, 20)), 1) == 1)
        
    def __len__(self):
        return len(self.df)

    def norm(self, pos):
        ref = pos.flatten()
        ref = ref[~ref.isnan()]
        mu, std = ref.mean(), ref.std()
        return (pos - mu) / std

    def get_pos(self, row):
        # pos = np.load(row.path)
        pos = npy[row.idx:row.idx + row.length].copy()
        if self.phase == "train":
            pos = aug2(pos)
            pos[:,-42:-21] = random_hand_op_h4(pos[:,-42:-21])
            pos[:,-21:] = random_hand_op_h4(pos[:,-21:])
        pos = torch.tensor(pos.astype(np.float32))
        # lip, lhand, rhand = pos[:,LIP], pos[:,468:489], pos[:,522:543]
        lip, lhand, rhand = pos[:,:-42], pos[:,-42:-21], pos[:,-21:]
        if self.phase == "train":
            if np.random.rand() < 0.5:
                lhand, rhand = rhand, lhand
                lhand[...,0] *= -1; rhand[...,0] *= -1
                lip[:,4:22], lip[:,22:40] = lip[:,22:40], lip[:,4:22]
                lip[...,0] *= -1
        lhand = lhand if lhand.isnan().sum() < rhand.isnan().sum() else flip_hand(lip, rhand)

        pos = self.norm(torch.cat([lip, lhand], 1))

        offset = torch.zeros_like(pos[-1:])
        movement = pos[:-1] - pos[1:]
        dpos = torch.cat([movement, offset])
        rdpos = torch.cat([offset, -movement])

        ld = torch.linalg.vector_norm(lhand[:,self.dis_idx0,:2] - lhand[:,self.dis_idx1,:2], dim = -1)
        lipd = torch.linalg.vector_norm(lip[:,self.dis_idx2,:2] - lip[:,self.dis_idx3,:2], dim = -1)

        lsim = F.cosine_similarity(lhand[:,HAND_ANGLES[:,0]] - lhand[:,HAND_ANGLES[:,1]],
                                   lhand[:,HAND_ANGLES[:,2]] - lhand[:,HAND_ANGLES[:,1]], -1)
        lipsim = F.cosine_similarity(lip[:,LIP_ANGLES[:,0]] - lip[:,LIP_ANGLES[:,1]],
                                     lip[:,LIP_ANGLES[:,2]] - lip[:,LIP_ANGLES[:,1]], -1)
        pos = torch.cat([
            pos.flatten(1),
            dpos.flatten(1),
            rdpos.flatten(1),
            lipd.flatten(1),
            ld.flatten(1),
            lipsim.flatten(1),
            lsim.flatten(1),
        ], -1)
        pos = torch.where(torch.isnan(pos), torch.tensor(0.0, dtype = torch.float32).to(pos), pos)
        if len(pos) > self.max_len:
            pos = pos[np.linspace(0, len(pos), self.max_len, endpoint = False)]
        return pos

    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        pos = self.get_pos(row)
        attention_mask = torch.zeros(self.max_len)
        attention_mask[:len(pos)] = 1
        pos = torch.cat([pos, torch.zeros((self.max_len - pos.shape[0], pos.shape[1]), dtype = pos.dtype)], 0)
        return {"inputs_embeds": pos, "attention_mask": attention_mask}, int(row.label)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = cfg.config
        self.model = transformers.AutoModel.from_config(config)
        self.emb = nn.Linear(config.pre_emb_size, config.hidden_size, bias = False)
        self.cls_emb = nn.Parameter(torch.zeros((1, 1, config.hidden_size)))
        self.fc = nn.Sequential(
            nn.Dropout(cfg.drop_rate),
            nn.Linear(config.hidden_size, config.num_labels)
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing = cfg.label_smooth)
        self.change_layer(self.model, config)
        self.model.post_init()

        # if hasattr(cfg, "pretrained_path"):
        #     stt = torch.load(cfg.pretrained_path, map_location = "cpu")["state_dict"]
        #     stt = {k[6:]: v for k, v in stt.items()}
        #     if any([k.startswith("student.") for k in stt]):
        #         stt = {k[8:]: v for k, v in stt.items() if k.startswith('student.')}
        #     self.load_state_dict(stt, strict = False)

    def change_layer(self, module, config):
        for n, m in module.named_children():
            if m.__class__.__name__ in ["BertEncoder", "DebertaV2Encoder"]:
                enc = eval(m.__class__.__name__ + "CPE")
                module.add_module(n, enc(config))
            else:
                self.change_layer(m, config)


    def embed(self, inputs):
        inputs_embeds = self.emb(inputs["inputs_embeds"])
        inputs_embeds = torch.cat([self.cls_emb.repeat(inputs_embeds.shape[0], 1, 1), inputs_embeds], 1)
        attention_mask = torch.cat([torch.ones((inputs_embeds.shape[0], 1)).to(inputs_embeds), inputs["attention_mask"]], 1) if "attention_mask" in inputs else None
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    def classifier(self, out, label = None):
        out = self.fc(out)
        if label is not None:
            loss = self.criterion(out, label)
            return out, loss
        else:
            return out

    def forward(self, inputs, label = None):
        inputs = self.embed(inputs)
        out = self.model(**inputs)["last_hidden_state"][:,0]
        return self.classifier(out, label = label)

class LightModel(ptl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Model(cfg)
        self.save_hyperparameters({
            **{k: v for k, v in cfg.__dict__.items() if not k.startswith("__") and k not in {"split", "config"}},
            "config": {k: v for k, v in cfg.config.to_diff_dict().items() if k not in {"id2label", "label2id"}},
        })
        
    def prepare_data(self):
        df_train = df.loc[self.cfg.split[self.cfg.fold][0]]
        df_valid = df.loc[self.cfg.split[self.cfg.fold][1]]
        self.dl_train = DataLoader(Data(df_train, self.cfg, "train"), batch_size = self.cfg.batch_size, shuffle = True, num_workers = 8, drop_last = True)
        self.dl_valid = DataLoader(Data(df_valid, self.cfg, "valid"), batch_size = self.cfg.batch_size, shuffle = False, num_workers = 8, drop_last = False)
    
    def train_dataloader(self):
        return self.dl_train
    
    def val_dataloader(self):
        return self.dl_valid
    
    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr = self.cfg.learning_rate, k = 5, alpha = 0.5)
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(optimizer, self.cfg.warmup_epochs, self.cfg.num_epochs, warmup_start_lr = self.cfg.learning_rate)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.cfg.warmup_epochs, self.cfg.plateau_epochs, self.cfg.num_epochs, warmup_start_lr = 1e-6)
        return [optimizer], [scheduler]
    
    def forward(self, x, y):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat, loss = self(x, y)
        self.log("train_loss", loss)
        self.log("total_steps", len(self.train_dataloader()))
        self.log("num_epochs", self.cfg.num_epochs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat, loss = self(x, y)
        self.log("valid_loss", loss, prog_bar = True)
        return y, yhat
    
    def validation_step_end(self, output):
        return output
    
    def validation_epoch_end(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).detach().cpu().numpy()
        for k in self.cfg.topks:
            self.log(f"valid_top{k}", 
                     top_k_accuracy_score(y, yhat, k = k, labels = list(range(self.cfg.config.num_labels))), 
                     prog_bar = True)



config = transformers.BertConfig()
config.hidden_size = 256
config.intermediate_size = 512 # config.hidden_size // 2
config.num_attention_heads = 4
config.max_position_embeddings = 96 + 1
config.num_hidden_layers = 4
config.vocab_size = 1
config.pre_emb_size = 972
config.num_labels = 250
config.output_hidden_states = True
config.hidden_act = "silu"
config.cpe_kernel_size = 3
config.cpe_start = 1
config.cpe_end = 4

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type = int, default = 0)
fold = parser.parse_args().fold
fold = 1
class cfg:
    fold = fold
    split = list(StratifiedKFold(8, random_state = 42, shuffle = True).split(df, y = df.label))
    num_epochs = 70
    warmup_epochs = 45
    # plateau_epochs = 45
    learning_rate = 2e-4
    name = "rsplit"
    version = f"N1"
    monitor = "valid_top1"
    batch_size = 128
    topks = [1]
    config = config
    log_step = 100
    drop_rate = 0.1
    label_smooth = 0.50  
    # pretrained_path = "./logs/folds/d_a2h4_d2d2a4t_L3M1E60/f0/epoch=58_valid_top1=0.717.ckpt"
    
if __name__ == "__main__":
    logger = []; callbacks = []
    logger = [LogJsonLogger("./logs", name = cfg.name, version = cfg.version, flush_logs_every_n_steps = cfg.log_step), WandbLogger(name = cfg.version, project = cfg.name, save_dir = f"./logs")]
    callbacks = [ModelCheckpoint(dirpath = os.path.join("./logs", cfg.name, cfg.version), save_top_k = 5, filename = '{epoch}_{' + cfg.monitor + ':.3f}', save_last = True, save_weights_only = True, mode = "max", monitor = cfg.monitor), TQDMProgressBar(refresh_rate = 20), LearningRateMonitor()]

    trainer = ptl.Trainer(
        accelerator = "gpu",
        devices = [0],
        precision = 16,
        gradient_clip_val = 10,
        max_epochs = cfg.num_epochs,
        logger = logger,
        callbacks = callbacks,
        log_every_n_steps = cfg.log_step
    )
    model = LightModel(cfg)
    trainer.fit(model)