import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pack_seq(seq, max_length=384):

    length = [min(len(s), max_length) for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[1]
    L = max(length)

    x = torch.zeros((batch_size, L, K, 3)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l]
        x_mask[b, l:] = 1
    x_mask = (x_mask>0.5)
    x = x.reshape(batch_size, -1, K*3)
    return x, x_mask


class HardSwish(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return x * F.relu6(x+3) * 0.16666667


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head, batch_first):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x, x, x, key_padding_mask=x_mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_head, out_dim, batch_first=True):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_head, batch_first)
        self.ffn = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


def positional_encoding(length, embed_dim):
    dim = embed_dim//2
    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)
    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)
    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        max_length = 384
        num_point = 80
        embed_dim = 512

        num_head = 8
        num_block = 1

        pos_embed = positional_encoding(max_length, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed)

        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))
        self.x_embed = nn.Sequential(
            nn.Linear(num_point * 3, embed_dim, bias=False),
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_head,
                embed_dim,
            ) for i in range(num_block)
        ])

    def forward(self, x, x_mask):
        # x, x_mask = pack_seq(xyz)

        B, L, _ = x.shape
        x = self.x_embed(x)
        x = x + self.pos_embed[:L].unsqueeze(0)

        x = torch.cat([self.cls_embed.unsqueeze(0).repeat(B, 1, 1), x], 1)
        x_mask = torch.cat([
            torch.zeros(B, 1).to(x_mask),
            x_mask
        ], 1)

        for block in self.encoder:
            x = block(x, x_mask)

        cls_ = x[:, 0]
        return cls_


class TREFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #self.eff = EfficientNet.from_pretrained('efficientnet-b0', num_classes=512, dropout_rate=cfg.drop_rate)
        self.transformer = TransformerNet()

        self.logits = nn.Sequential(
                                    nn.Linear(512, 512),  # 1024
                                    nn.ReLU(),
                                    nn.Dropout(cfg.drop_rate),
                                    nn.Linear(512, 250))

    def forward(self, img, tr, tr_mask):
        #x1 = self.eff(img)
        x2 = self.transformer(tr, tr_mask)

        # x = x1 + x2
        #x = torch.cat([x1, x2], dim=1)
        x = self.logits(x2)
        return x


def do_random_affine(xyz, scale=(0.8, 1.3), shift=(-0.08, 0.08), degree=(-16, 16)):
    if scale is not None:
        scale = np.random.uniform(*scale)
        xyz = scale*xyz

    if shift is not None:
        shift = np.random.uniform(*shift)
        xyz = xyz + shift

    if degree is not None:
        degree = np.random.uniform(*degree)
        radian = degree/180*np.pi
        c = np.cos(radian)
        s = np.sin(radian)
        rotate = np.array([
            [c,-s],
            [s, c],
        ]).T
        xyz[..., :2] = xyz[..., :2] @rotate

    return xyz


class DatasetTrEfV1(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291][::2][1:]
        lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][::2][1:]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][::2][1:]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][::2][1:]

        LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
        pose = [489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                507, 508, 509, 510, 511, 512]

        l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                  481, 482, 483, 484, 485, 486, 487, 488]
        r_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                  533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + l_hand + pose + r_hand)

        self.max_len = 384

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        yy = self.df[i]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)
        yy = yy[:, self.interesting_idx, :]

        yy_img = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        if yy.shape[0] >= self.max_len:
            yy_trans = F.interpolate(yy[None, None, :], size=(self.max_len, self.new_size[1], 3), mode='nearest')[0, 0]
        else:
            yy_trans = yy

        yy_img = yy_img.permute(2, 0, 1)

        return yy_img, yy_trans, self.Y[i]


class DatasetTrEfV2(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
                 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518,
                 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + other)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        yy = self.df[i]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)
        yy = yy[:, self.interesting_idx, :]

        yy_img = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy_img = yy_img.permute(2, 0, 1)

        return yy_img, yy, self.Y[i]


class DatasetTrEfV3(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
                 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518,
                 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + other)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        yy = self.df[i]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)
        yy = yy[:, self.interesting_idx, :]

        yy_img = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy_trans = F.interpolate(yy[None, None, :], size=(192, 112, 3), mode='nearest')[0, 0]

        yy_img = yy_img.permute(2, 0, 1)

        return yy_img, yy_trans, self.Y[i]


def tref_collate(batch):
    """
    collate function is necessary for transferring data into GPU
    :param batch: Input tensor
    :return tuple with labels and batch tensors
    """
    img = torch.stack([x[0] for x in batch])
    tr_x, tr_mask = pack_seq([x[1] for x in batch])
    label = torch.tensor([x[2] for x in batch]).long()

    return img, tr_x, tr_mask, label
