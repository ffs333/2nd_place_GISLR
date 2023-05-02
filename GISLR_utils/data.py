import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchaudio

from .eff_transormer import DatasetTrEfV1, tref_collate, DatasetTrEfV2, DatasetTrEfV3
from .mixup_data import DatasetImageSmall80Mixup, DatasetImageSmall112Mixup, DatasetImageSmall120Mixup
from .partial_aug_data import DatasetImageSmall80Partial


def prepare_loaders(_cfg, folds, fold):
    """
    Prepare and build train and eval data loaders
    """
    train_inds = folds[folds.fold != fold].index
    valid_inds = folds[folds.fold == fold].index

    if fold == 0 and _cfg.drop_bad_inds:
        bad_inds = np.load(_cfg.base_path + 'bad_indsf0.npy')
        train_inds = [x for x in train_inds if x not in bad_inds]

    if _cfg.COLAB:
        import os
        if not (os.path.exists('/content/data.npy') and os.path.exists('/content/Y.npy')):
            import shutil
            shutil.copyfile(_cfg.base_path + 'gen_xyz/Y.npy', '/content/Y.npy')
            shutil.copyfile(_cfg.base_path + 'gen_xyz/data.npy', '/content/data.npy')
        data = np.load('/content/data.npy', allow_pickle=True)
        Y = np.load('/content/Y.npy')
    else:
        data = np.load(_cfg.base_path + 'gen_xyz/data.npy', allow_pickle=True)
        Y = np.load(_cfg.base_path + 'gen_xyz/Y.npy')

    if _cfg.deal_with_len:
        print(f'Len train_inds before: {len(train_inds)}')
        print(f'Len valid_inds before: {len(valid_inds)}')
        if _cfg.moreN:
            print(f'Train long >8 len')
            new_train_inds, new_valid_inds = [], []
            for ind in train_inds:
                if data[ind].shape[0] > 8:
                    new_train_inds.append(ind)
            train_inds = np.array(new_train_inds)
            for ind in valid_inds:
                if data[ind].shape[0] > 8:
                    new_valid_inds.append(ind)
            valid_inds = np.array(new_valid_inds)
        else:
            print(f'Multiply shorts')
            add_train_inds, new_valid_inds = [], []
            for ind in train_inds:
                if data[ind].shape[0] < 9:
                    for _ in range(7):
                        add_train_inds.append(ind)
                elif 9 <= data[ind].shape[0] < 13:
                    for _ in range(3):
                        add_train_inds.append(ind)
            train_inds = np.concatenate([train_inds, np.array(add_train_inds)])

            for ind in valid_inds:
                if data[ind].shape[0] <= 10:
                    new_valid_inds.append(ind)
            valid_inds = np.array(new_valid_inds)
        print(f'Len train_inds after: {len(train_inds)}')
        print(f'Len valid_inds after: {len(valid_inds)}')

    train_data = [data[i] for i in train_inds]
    valid_data = [data[i] for i in valid_inds]

    Y_train = Y[train_inds]
    Y_valid = Y[valid_inds]

    print(f'Size of train dataset: {len(train_inds)}')
    print(f'Size of valid dataset: {len(valid_inds)}')

    if _cfg.dataset == 'img_80':
        train_dataset = DatasetImageSmall80(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        valid_dataset = DatasetImageSmall80(cfg=_cfg, df=valid_data, Y=Y_valid, train_mode=False)
        collate_fn = None

    elif _cfg.dataset == 'img_80_norm':
        train_dataset = DatasetImageSmall80Normed(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        valid_dataset = DatasetImageSmall80Normed(cfg=_cfg, df=valid_data, Y=Y_valid, train_mode=False)
        collate_fn = None

    elif _cfg.dataset == 'img_80_mixup':
        train_dataset = DatasetImageSmall80Mixup(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        valid_dataset = DatasetImageSmall80Mixup(cfg=_cfg, df=valid_data, Y=Y_valid, train_mode=False)
        collate_fn = None
    elif _cfg.dataset == 'img_80_partial':
        train_dataset = DatasetImageSmall80Partial(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        valid_dataset = DatasetImageSmall80Partial(cfg=_cfg, df=valid_data, Y=Y_valid, train_mode=False)
        collate_fn = None
    elif _cfg.dataset == 'img_120_mixup':
        train_dataset = DatasetImageSmall120Mixup(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        valid_dataset = DatasetImageSmall120Mixup(cfg=_cfg, df=valid_data, Y=Y_valid, train_mode=False)
        collate_fn = None

    else:
        raise ValueError('Error in "prepare_loaders" function:',
                         f'Wrong dataset name. Choose one from ["v1"] ')

    train_loader = DataLoader(train_dataset,
                              batch_size=_cfg.train_bs,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=_cfg.valid_bs,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader, valid_inds


def prepare_loader_full(_cfg, folds):
    """
    Prepare and build train and eval data loaders
    """

    print(f'Size of train dataset: {len(folds)}')

    train_inds = folds.index

    if _cfg.COLAB:
        import os
        if not (os.path.exists('/content/data.npy') and os.path.exists('/content/Y.npy')):
            import shutil
            shutil.copyfile(_cfg.base_path + 'gen_xyz/Y.npy', '/content/Y.npy')
            shutil.copyfile(_cfg.base_path + 'gen_xyz/data.npy', '/content/data.npy')
        data = np.load('/content/data.npy', allow_pickle=True)
        Y = np.load('/content/Y.npy')
    else:
        data = np.load(_cfg.base_path + 'gen_xyz/data.npy', allow_pickle=True)
        Y = np.load(_cfg.base_path + 'gen_xyz/Y.npy')

    train_data = data
    Y_train = Y

    print(f'Size of train dataset: {len(data)}')

    if _cfg.dataset == 'img_80_mixup':
        train_dataset = DatasetImageSmall80Mixup(cfg=_cfg, df=train_data, Y=Y_train, train_mode=True)
        collate_fn = None
    else:
        raise ValueError('Error in "prepare_loaders" function:',
                         f'Wrong dataset name. Choose one from [] ')

    train_loader = DataLoader(train_dataset,
                              batch_size=_cfg.train_bs,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)

    return train_loader


class DatasetV1(torch.utils.data.Dataset):
    def __init__(self, X, X_inv, XM, XM_inv, y, aug_prob=0.4):
        self.X = X
        self.X_inv = X_inv

        self.XM = XM
        self.XM_inv = XM_inv

        self.y = y
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if random.random() > self.aug_prob:  # 0.33
            x = self.X[i]
            xm = self.XM[i]
        else:
            x = self.X_inv[i]
            xm = self.XM_inv[i]

        return x.float(), xm.astype(np.float32), self.y[i]


def do_random_affine(xyz, scale=(0.8, 1.3), shift=(-0.08, 0.08), degree=(-16, 16), p=0.5):
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


class DatasetImageSmall80Normed(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob
        self.img_mask = cfg.img_masking

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
        r_hand =[522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + l_hand + pose + r_hand)

        # norm params
        self.x_mean = torch.tensor(0.48282188177108765).float()
        self.y_mean = torch.tensor(0.6022918224334717).float()
        self.z_mean = torch.tensor(-0.4806228578090668).float()
        self.x_std = torch.tensor(0.22193588316440582).float()
        self.y_std = torch.tensor(0.252733051776886).float()
        self.z_std = torch.tensor(0.68902188539505).float()

        # lips
        self.x_mean_lips = torch.tensor(0.47860637307167053).float()
        self.x_std_lips = torch.tensor(0.032161612063646317).float()
        self.y_mean_lips = torch.tensor(0.4729398488998413).float()
        self.y_std_lips = torch.tensor(0.01253026258200407).float()
        self.z_mean_lips = torch.tensor(-0.025047844275832176).float()
        self.z_std_lips = torch.tensor(0.010506421327590942).float()

        # pose
        self.x_mean_pose = torch.tensor(0.5086792707443237).float()
        self.y_mean_pose = torch.tensor(0.741327166557312).float()
        self.z_mean_pose = torch.tensor(-1.1642873287200928).float()
        self.x_std_pose = torch.tensor(0.30861276388168335).float()
        self.y_std_pose = torch.tensor(0.33899742364883423).float()
        self.z_std_pose = torch.tensor(0.6623932123184204).float()

        # right hand
        self.x_mean_right = torch.tensor(0.32876524329185486).float()
        self.x_std_right = torch.tensor(0.0852568969130516).float()
        self.y_mean_right = torch.tensor(0.5775985717773438).float()
        self.y_std_right = torch.tensor(0.08005094528198242).float()
        self.z_mean_right = torch.tensor(-0.06172217056155205).float()
        self.z_std_right = torch.tensor(0.045487865805625916).float()

        # Left hand
        self.x_mean_left = torch.tensor(0.6159984469413757).float()
        self.x_std_left = torch.tensor(0.08954811096191406).float()
        self.y_mean_left = torch.tensor(0.5661182403564453).float()
        self.y_std_left = torch.tensor(0.08524303883314133).float()
        self.z_mean_left = torch.tensor(-0.05598198249936104).float()
        self.z_std_left = torch.tensor(0.045476797968149185).float()

        self.lips_ar = np.arange(0, 18)
        self.l_hand = np.arange(18, 39)
        self.pose_ar = np.arange(39, 59)
        self.r_hand = np.arange(59, 80)

        self.fill_na = torch.tensor(0.).float()
        self.freq_m = torchaudio.transforms.FrequencyMasking(cfg.freq_m)  # 10
        self.time_m = torchaudio.transforms.TimeMasking(cfg.time_m)  # 16

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):
        # sample = self.df.loc[i]
        # yy = load_relevant_data_subset(self.base_dir + f'asl-signs/' + sample['path'])
        # yy = load_relevant_data_subset(self.base_dir + sample['path'])
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        # yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        x_lips = torch.where(~torch.isnan(yy[:, self.lips_ar, 0]),
                             (yy[:, self.lips_ar, 0] - self.x_mean_lips) / (4. * self.x_std_lips), self.fill_na)
        x_lhand = torch.where(~torch.isnan(yy[:, self.l_hand, 0]),
                              (yy[:, self.l_hand, 0] - self.x_mean_left) / (5. * self.x_std_left), self.fill_na)
        x_pose = torch.where(~torch.isnan(yy[:, self.pose_ar, 0]),
                             (yy[:, self.pose_ar, 0] - self.x_mean_pose) / (4. * self.x_std_pose), self.fill_na)
        x_rhand = torch.where(~torch.isnan(yy[:, self.r_hand, 0]),
                              (yy[:, self.r_hand, 0] - self.x_mean_right) / (5. * self.x_std_right), self.fill_na)
        new_x = torch.cat([x_lips, x_lhand, x_pose, x_rhand], dim=1)

        y_lips = torch.where(~torch.isnan(yy[:, self.lips_ar, 1]),
                             (yy[:, self.lips_ar, 1] - self.y_mean_lips) / (4. * self.y_std_lips), self.fill_na)
        y_lhand = torch.where(~torch.isnan(yy[:, self.l_hand, 1]),
                              (yy[:, self.l_hand, 1] - self.y_mean_left) / (5. * self.y_std_left), self.fill_na)
        y_pose = torch.where(~torch.isnan(yy[:, self.pose_ar, 1]),
                             (yy[:, self.pose_ar, 1] - self.y_mean_pose) / (4. * self.y_std_pose), self.fill_na)
        y_rhand = torch.where(~torch.isnan(yy[:, self.r_hand, 1]),
                              (yy[:, self.r_hand, 1] - self.y_mean_right) / (5. * self.y_std_right), self.fill_na)
        new_y = torch.cat([y_lips, y_lhand, y_pose, y_rhand], dim=1)

        z_lips = torch.where(~torch.isnan(yy[:, self.lips_ar, 2]),
                             (yy[:, self.lips_ar, 2] - self.z_mean_lips) / (4. * self.z_std_lips), self.fill_na)
        z_lhand = torch.where(~torch.isnan(yy[:, self.l_hand, 2]),
                              (yy[:, self.l_hand, 2] - self.z_mean_left) / (5. * self.z_std_left), self.fill_na)
        z_pose = torch.where(~torch.isnan(yy[:, self.pose_ar, 2]),
                             (yy[:, self.pose_ar, 2] - self.z_mean_pose) / (4. * self.z_std_pose), self.fill_na)
        z_rhand = torch.where(~torch.isnan(yy[:, self.r_hand, 2]),
                              (yy[:, self.r_hand, 2] - self.z_mean_right) / (5. * self.z_std_right), self.fill_na)
        new_z = torch.cat([z_lips, z_lhand, z_pose, z_rhand], dim=1)

        yy = torch.stack([new_x, new_y, new_z], dim=0)

        if random.random() < self.img_mask and self.train_mode:
            yy = self.time_m(self.freq_m(yy.unsqueeze(0)))[0]

        return yy, meta, self.Y[i]


class DatasetImageV1_112(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size #(112, 112, 3)  # (128, 128, 3)

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
                 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
                 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + other)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, self.Y[i]


class DatasetImageV1(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size #(112, 112, 3)  # (128, 128, 3)

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other_sv = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
                 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
                 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

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

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):
        # sample = self.df.loc[i]
        # yy = load_relevant_data_subset(self.base_dir + f'asl-signs/' + sample['path'])
        # yy = load_relevant_data_subset(self.base_dir + sample['path'])
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]


def center_data_nan2(tensor):
    centered_tensors = []

    for dim in range(tensor.shape[2]):
        # Создание маски для игнорирования значений nan
        mask = ~torch.isnan(tensor[:, :, dim])

        # Вычисление среднего значения, игнорируя nan
        mean_val = torch.mean(torch.masked_select(tensor[:, :, dim], mask))

        # Вычитание среднего значения, игнорируя nan
        centered_dim = tensor[:, :, dim] - mean_val
        centered_dim_with_nan = torch.where(mask, centered_dim, tensor[:, :, dim])

        centered_tensors.append(centered_dim_with_nan)

    centered_tensor = torch.stack(centered_tensors, dim=-1)
    return centered_tensor


def flip_pose(points):
    l_hand_indices = torch.tensor(np.arange(18, 39))
    pose_ind1_l, pose_ind1_r = torch.tensor(np.array([40, 41, 44]).astype(int)), torch.tensor(
        np.array([42, 43, 45]).astype(int))
    pose_ind2_l, pose_ind2_r = torch.tensor(np.array([46, 48, 50, 52, 54, 56]).astype(int)), torch.tensor(
        np.array([47, 49, 51, 53, 55, 57]).astype(int))
    r_hand_indices = torch.tensor(np.arange(59, 80))
    flipped_points = points.clone()

    x_max = flipped_points[:, :, 0][torch.where(~torch.isnan(flipped_points[:, :, 0]))].max()

    flipped_points[:, l_hand_indices] = points[:, r_hand_indices]
    flipped_points[:, r_hand_indices] = points[:, l_hand_indices]

    # Отражение координат x
    flipped_points[:, pose_ind1_l] = points[:, pose_ind1_r]
    flipped_points[:, pose_ind1_r] = points[:, pose_ind1_l]

    flipped_points[:, pose_ind2_l] = points[:, pose_ind2_r]
    flipped_points[:, pose_ind2_r] = points[:, pose_ind2_l]

    flipped_points[:, :, 0] = x_max - flipped_points[:, :, 0]

    return flipped_points


def scale_parts(points):
    scaled_points = points.clone()
    lips_indices = np.arange(0, 18)
    l_hand_indices = np.arange(18, 39)
    pose_indices = np.arange(39, 59)
    r_hand_indices = np.arange(59, 80)

    def apply_scale(part_points, scale):
        part_center = torch.nanmean(part_points, dim=1, keepdim=True)
        return part_center + (part_points - part_center) * scale

    lips_scale = np.random.randint(88, 115) / 100.
    l_hand_scale = np.random.randint(88, 115) / 100.
    r_hand_scale = np.random.randint(88, 115) / 100.
    pose_scale = np.random.randint(88, 115) / 100.

    scaled_points[:, lips_indices] = apply_scale(points[:, lips_indices], lips_scale)
    scaled_points[:, l_hand_indices] = apply_scale(points[:, l_hand_indices], l_hand_scale)
    scaled_points[:, r_hand_indices] = apply_scale(points[:, r_hand_indices], r_hand_scale)
    scaled_points[:, pose_indices] = apply_scale(points[:, pose_indices], pose_scale)

    return scaled_points


class DatasetImageSmall80(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob
        self.scale_prob = cfg.scale_prob
        self.img_mask = cfg.img_masking
        self.shift_prob = cfg.shift_prob

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
        r_hand =[522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + l_hand + pose + r_hand)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

        self.hands__ = np.concatenate([np.arange(self.hand_left[0], self.hand_left[1]),
                                       np.arange(self.hand_right[0], self.hand_right[1])])

        self.freq_m = torchaudio.transforms.FrequencyMasking(cfg.freq_m)  # 10
        self.time_m = torchaudio.transforms.TimeMasking(cfg.time_m)  # 16

        # lips
        self.x_std_lips = torch.tensor(0.032161612063646317).float()
        self.y_std_lips = torch.tensor(0.01253026258200407).float()
        self.z_std_lips = torch.tensor(0.010506421327590942).float()

        # pose
        self.x_std_pose = torch.tensor(0.30861276388168335).float()
        self.y_std_pose = torch.tensor(0.33899742364883423).float()
        self.z_std_pose = torch.tensor(0.6623932123184204).float()

        # right hand
        self.x_std_right = torch.tensor(0.0852568969130516).float()
        self.y_std_right = torch.tensor(0.08005094528198242).float()
        self.z_std_right = torch.tensor(0.045487865805625916).float()

        # Left hand
        self.x_std_left = torch.tensor(0.08954811096191406).float()
        self.y_std_left = torch.tensor(0.08524303883314133).float()
        self.z_std_left = torch.tensor(0.045476797968149185).float()

        self.lips_ar = np.arange(0, 18)
        self.l_hand = np.arange(18, 39)
        self.pose_ar = np.arange(39, 59)
        self.r_hand = np.arange(59, 80)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def rand_float10(a_min, a_max):
        return np.random.randint(a_min, a_max) / 10.

    def shift_aug(self, points):
        points[:, self.lips_ar, 0] += (self.x_std_lips/self.rand_float10(60, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.lips_ar, 1] += (self.y_std_lips/self.rand_float10(80, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.lips_ar, 2] += (self.z_std_lips/self.rand_float10(120, 200))*(1 if random.random() > 0.5 else -1)

        points[:, self.r_hand, 0] += (self.x_std_right/self.rand_float10(60, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.r_hand, 1] += (self.y_std_right/self.rand_float10(80, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.r_hand, 2] += (self.z_std_right/self.rand_float10(120, 200))*(1 if random.random() > 0.5 else -1)

        points[:, self.pose_ar, 0] += (self.x_std_pose/self.rand_float10(110, 200))*(1 if random.random() > 0.5 else -1)
        points[:, self.pose_ar, 1] += (self.y_std_pose/self.rand_float10(110, 200))*(1 if random.random() > 0.5 else -1)
        points[:, self.pose_ar, 2] += (self.z_std_pose/self.rand_float10(160, 250))*(1 if random.random() > 0.5 else -1)

        points[:, self.l_hand, 0] += (self.x_std_left/self.rand_float10(60, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.l_hand, 1] += (self.y_std_left/self.rand_float10(80, 150))*(1 if random.random() > 0.5 else -1)
        points[:, self.l_hand, 2] += (self.z_std_left/self.rand_float10(120, 200))*(1 if random.random() > 0.5 else -1)

        return points

    def __getitem__(self, i):
        # sample = self.df.loc[i]
        # yy = load_relevant_data_subset(self.base_dir + f'asl-signs/' + sample['path'])
        # yy = load_relevant_data_subset(self.base_dir + sample['path'])
        yy = self.df[i]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)

        yy = torch.tensor(yy)

        yy = yy[:, self.interesting_idx, :]

        if self.train_mode:
            if random.random() < self.invert_prob:
                yy = flip_pose(yy)
            if random.random() < self.shift_prob:
                yy = self.shift_aug(yy)
            if random.random() < self.scale_prob:
                yy = scale_parts(yy)

        yy = torch.nan_to_num(yy, 0.0)

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest') # this is base

        if random.random() < self.img_mask and self.train_mode:
            yy = self.time_m(self.freq_m(yy.squeeze(0).permute(0, 3, 1, 2)))[0]
        else:
            yy = yy[0, 0]
            yy = yy.permute(2, 0, 1)

        return yy, self.Y[i]


class DatasetImageSmall88(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = (88, 88, 3)

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        lipsUpperOuter = [61, 185, 40, 37, 0, 267, 270, 409, 291]
        lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][::2]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][::2]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][::2]

        LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
        pose = [489, 490, 492, 493, 494, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                507, 508, 509, 510, 511, 512]

        l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                 481, 482, 483, 484, 485, 486, 487, 488]
        r_hand =[522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.interesting_idx = np.array(LIPS + l_hand + pose + r_hand)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):
        # sample = self.df.loc[i]
        # yy = load_relevant_data_subset(self.base_dir + f'asl-signs/' + sample['path'])
        # yy = load_relevant_data_subset(self.base_dir + sample['path'])
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]


class DatasetImageV3(torch.utils.data.Dataset):  # drop legs from pose
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size #(112, 112, 3)  # (128, 128, 3)

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                 481, 482, 483, 484, 485, 486, 487, 488,  # left hand
                 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,  # pose
                 507, 508, 509, 510, 511,  # pose
                 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]  # right hand

        self.interesting_idx = np.array(LIPS + other)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):
        # sample = self.df.loc[i]
        # yy = load_relevant_data_subset(self.base_dir + f'asl-signs/' + sample['path'])
        # yy = load_relevant_data_subset(self.base_dir + sample['path'])
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]


class DatasetImageV4(torch.utils.data.Dataset):  # same with 3 and drop full nan hands
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                 481, 482, 483, 484, 485, 486, 487, 488,  # left hand
                 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,  # pose
                 507, 508, 509, 510, 511,  # pose
                 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]  # right hand

        self.handss = np.array([468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                           481, 482, 483, 484, 485, 486, 487, 488,
                           522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                           533, 534, 535, 536, 537, 538, 539, 540, 541, 542])

        self.interesting_idx = np.array(LIPS + other)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):

        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = yy[~torch.all(torch.isnan(yy[:, self.handss, 0]), dim=1), :]
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]


class DatasetImageV6(torch.utils.data.Dataset):  # BOOMERANG
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        other = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                 481, 482, 483, 484, 485, 486, 487, 488,  # left hand
                 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,  # pose
                 507, 508, 509, 510, 511,  # pose
                 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                 533, 534, 535, 536, 537, 538, 539, 540, 541, 542]  # right hand

        self.handss = np.array([468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                           481, 482, 483, 484, 485, 486, 487, 488,
                           522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                           533, 534, 535, 536, 537, 538, 539, 540, 541, 542])

        self.interesting_idx = np.array(LIPS + other)

        self.lips = np.array(LIPS)
        self.hand_left = np.arange(468, 489)
        self.hand_right = np.arange(522, 543)

        self.pad_shapes = torch.tensor([7, 14, 28, 56])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_flat_features(x, idx_range):
        len_range = len(idx_range)
        return x[:, idx_range, :].contiguous().view(-1, len_range * 3)

    def __getitem__(self, i):

        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)
        yy = torch.nan_to_num(yy, 0.0)

        yy = yy[:, self.interesting_idx, :]

        if yy.shape[0] >= self.new_size[1] or yy.shape[0] == 1:
            yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]
        else:
            if yy.shape[0] < self.pad_shapes[-1]:
                inter_to = self.pad_shapes[torch.where(self.pad_shapes >= yy.shape[0])]
                # inter_to = self.pad_shapes[self.pad_shapes >= yy.shape[0]]

                yy = F.interpolate(yy[None, None, :], size=(inter_to[0], self.new_size[1], 3), mode='nearest')[0, 0]

                how_many_pad = (self.new_size[0] // yy.shape[0]) - 1

                yy_base = torch.clone(yy)
                for hm in range(how_many_pad):
                    yy_base = torch.flip(yy_base, dims=(0,))
                    yy = torch.cat([yy, yy_base], dim=0)

            else:
                yy = torch.cat([yy, torch.flip(yy, dims=(0,))])[:self.new_size[0]]
                yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]


class DatasetImageV5(torch.utils.data.Dataset):  # drop legs from pose and norm ot [-1, 1] every part and every axis
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size  # (112, 112, 3)  # (128, 128, 3)

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob

        self.train_mode = train_mode

        LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267,
                269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        left_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,  # left hand
                     481, 482, 483, 484, 485, 486, 487, 488]

        pose = [489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,  # pose
                507, 508, 509, 510, 511]

        right_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,  # right hand
                      533, 534, 535, 536, 537, 538, 539, 540, 541, 542]

        self.new_lips = np.arange(0, 40)
        self.new_left = np.arange(40, 61)
        self.new_pose = np.arange(61, 84)
        self.new_right = np.arange(84, 105)

        self.n_l0, self.n_l1 = self.new_left[0], self.new_left[-1]
        self.n_r0, self.n_r1 = self.new_right[0], self.new_right[-1]

        self.interesting_idx = np.array(LIPS + left_hand + pose + right_hand)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def __getitem__(self, i):
        yy = self.df[i]

        meta = len(yy) / 500.
        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)
            if random.random() < self.invert_prob:
                yy[:, :, 0] *= -1

        yy = torch.tensor(yy)

        yy = yy[:, self.interesting_idx, :]

        yy[:, self.new_lips, 0] /= torch.max(torch.abs(yy[:, self.new_lips, 0]))
        yy[:, self.new_lips, 1] /= torch.max(torch.abs(yy[:, self.new_lips, 1]))
        yy[:, self.new_lips, 2] /= torch.max(torch.abs(yy[:, self.new_lips, 2]))

        yy[:, self.new_pose, 0] /= torch.max(torch.abs(yy[:, self.new_pose, 0]))
        yy[:, self.new_pose, 1] /= torch.max(torch.abs(yy[:, self.new_pose, 1]))
        yy[:, self.new_pose, 2] /= torch.max(torch.abs(yy[:, self.new_pose, 2]))

        left_segs = self.consecutive(np.where(torch.all(~torch.all(yy[:, self.new_left].isnan(), dim=1), dim=1))[0])
        if len(left_segs[0]) != 0:
            for seg in left_segs:
                yy[seg, self.n_l0:self.n_l1, 0] /= torch.max(torch.abs(yy[seg, self.n_l0:self.n_l1, 0]))
                yy[seg, self.n_l0:self.n_l1, 1] /= torch.max(torch.abs(yy[seg, self.n_l0:self.n_l1, 1]))
                yy[seg, self.n_l0:self.n_l1, 2] /= torch.max(torch.abs(yy[seg, self.n_l0:self.n_l1, 2]))

        right_segs = self.consecutive(np.where(torch.all(~torch.all(yy[:, self.new_right].isnan(), dim=1), dim=1))[0])
        if len(right_segs[0]) != 0:
            for seg in right_segs:
                yy[seg, self.n_r0:self.n_r1, 0] /= torch.max(torch.abs(yy[seg, self.n_r0:self.n_r1, 0]))
                yy[seg, self.n_r0:self.n_r1, 1] /= torch.max(torch.abs(yy[seg, self.n_r0:self.n_r1, 1]))
                yy[seg, self.n_r0:self.n_r1, 2] /= torch.max(torch.abs(yy[seg, self.n_r0:self.n_r1, 2]))

        yy = torch.nan_to_num(yy, 0.)
        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')[0, 0]

        yy = yy.permute(2, 0, 1)

        return yy, meta, self.Y[i]
