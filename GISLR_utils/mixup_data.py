import math
import random

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchaudio
from scipy.interpolate import interp1d


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


def zero_out_random(tensor, percentage=0.15):
    num_to_zero_out = int(tensor.shape[1] * tensor.shape[2] * percentage)

    for i in range(tensor.shape[0]):
        indices = torch.randperm(tensor.shape[1] * tensor.shape[2])[:num_to_zero_out]

        for idx in indices:
            row = idx // tensor.shape[2]
            col = idx % tensor.shape[2]
            tensor[i, row, col] = 0

    return tensor


def random_rotation_around_axis(points, axis='x', min_angle=-15, max_angle=15):
    min_angle_rad = math.radians(min_angle)
    max_angle_rad = math.radians(max_angle)
    angle = random.uniform(min_angle_rad, max_angle_rad)

    if axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
    else:
        rotation_matrix = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])

    rotated_points = torch.matmul(points, rotation_matrix.T)

    return rotated_points


def random_interpolate(pos, scale=(0.8, 1.5), shift=(-0.5, 0.5)):
    scale = np.random.uniform(*scale)
    shift = np.random.uniform(*shift)
    orig_time = np.arange(len(pos))
    inte = interp1d(orig_time, pos, axis=0, fill_value="extrapolate")
    new_time = np.linspace(0 + shift, len(pos) - 1 + shift, int(round(len(pos) * scale)), endpoint=True)
    pos = inte(new_time).astype(np.float32)
    return torch.tensor(pos)


# yy = zero_out_random(yy, np.random.uniform(0.05, 0.2))


class DatasetImageSmall80Mixup(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob
        self.scale_prob = cfg.scale_prob
        self.img_mask = cfg.img_masking
        self.shift_prob = cfg.shift_prob
        self.zero_prob = cfg.zero_prob
        self.rotate_prob = cfg.rotate_prob
        self.replace_prob = cfg.replace_prob
        self.interpol_prob = cfg.interpol_prob
        self.tree_rot_prob = cfg.tree_rot_prob
        self.interp_nearest_random = cfg.interp_nearest_random

        self.mixup_prob = cfg.mixup_prob
        self.mixup_alpha = cfg.mixup_alpha

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

        self.parts_dict = {'lips': self.lips_ar,
                           'rh': self.r_hand,
                           'lh': self.l_hand,
                           'pose': self.pose_ar}

        self.use_normalize = cfg.normalize
        # nomalize
        # lips
        self.x_lips_adder = torch.tensor(0.47860637307167053).float()
        self.y_lips_adder = torch.tensor(0.4729398488998413).float()
        self.z_lips_adder = torch.tensor(-0.025047844275832176).float()

        # pose
        self.x_pose_adder = torch.tensor(0.5086792707443237).float()
        self.y_pose_adder = torch.tensor(0.741327166557312).float()
        self.z_pose_adder = torch.tensor(-1.1642873287200928).float()

        # rh
        self.x_right_adder = torch.tensor(0.32876524329185486).float()
        self.y_right_adder = torch.tensor(0.5775985717773438).float()
        self.z_right_adder = torch.tensor(-0.06172217056155205).float()

        # lh
        self.x_left_adder = torch.tensor(0.6159984469413757).float()
        self.y_left_adder = torch.tensor(0.5661182403564453).float()
        self.z_left_adder = torch.tensor(-0.05598198249936104).float()

        self.HAND_ROUTES = [
                            [0, *range(1, 5)],
                            [0, *range(5, 9)],
                            [0, *range(9, 13)],
                            [0, *range(13, 17)],
                            [0, *range(17, 21)],
                        ]
        self.HAND_TREES = sum([[np.array(route[i:]) for i in range(len(route) - 1)] for route in self.HAND_ROUTES], [])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def rotate_points(pos, center, alpha):
        radian = alpha / 180 * np.pi
        rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]])
        translated_points = (pos - center).reshape(-1, 2)
        rotated_points = np.dot(rotation_matrix, translated_points.T).T.reshape(*pos.shape)
        rotated_pos = rotated_points + center
        return rotated_pos

    def random_hand_rotate(self, pos, degree=(-4, 4), joint_prob=0.15):
        for tree in self.HAND_TREES:
            if np.random.rand() < joint_prob:
                alpha = np.random.uniform(*degree)
                center = pos[:, tree[0:1], :2]
                pos[:, tree[1:], :2] = self.rotate_points(pos[:, tree[1:], :2], center, alpha)
        return torch.tensor(pos, dtype=torch.float32)

    @staticmethod
    def norm(pos):
        ref = pos.flatten()
        ref = ref[~ref.isnan()]
        mu, std = ref.mean(), ref.std() + 1e-6
        return (pos - mu) / std

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

    def rotate_aug(self, img):
        ax = np.random.choice(['x', 'y', 'z'])  # 1 or 2

        parts = np.random.choice(['lips', 'rh', 'lh', 'pose'], np.random.randint(1, 5), replace=False)

        for part in parts:
            img[:, self.parts_dict[part], :] = random_rotation_around_axis(img[:, self.parts_dict[part], :],
                                                                           axis=ax,
                                                                           min_angle=-10, max_angle=10)

        return img

    def replace_aug(self, img, lab):
        parts = np.random.choice(['lips', np.random.choice(['rh', 'lh']), 'pose'],
                                 np.random.randint(1, 3), replace=False)

        for part in parts:
            repl_ind = np.random.choice(np.where(self.Y == lab)[0])
            repl, _ = self.get_one_item(repl_ind)

            if part in ['rh', 'lh']:
                if (img[:, :, :, self.parts_dict[part], :] == 0).all():
                    if random.random() < 0.93:
                        part_in = 'rh' if part == 'lh' else 'lh'
                    else:
                        part_in = part
                else:
                    part_in = part

                if (repl[:, :, :, self.parts_dict[part_in], :] == 0).all():
                    part_out = 'lh' if part_in == 'rh' else 'rh'
                    flipped_points = repl.clone()
                    x_max = flipped_points[:, :, 0].max()
                    flipped_points[:, :, :, :, 0] = x_max - flipped_points[:, :, :, :, 0]
                    flipped_points[:, :, :, self.parts_dict[part_in], :] = flipped_points[:, :, :, self.parts_dict[part_out], :]
                    repl = flipped_points
                part = part_in

            # rescale
            rescale = True
            if rescale:
                old_min0 = torch.min(repl[:, :, :, self.parts_dict[part], 0])
                old_max0 = torch.max(repl[:, :, :, self.parts_dict[part], 0])
                old_min1 = torch.min(repl[:, :, :, self.parts_dict[part], 1])
                old_max1 = torch.max(repl[:, :, :, self.parts_dict[part], 1])
                old_min2 = torch.min(repl[:, :, :, self.parts_dict[part], 2])
                old_max2 = torch.max(repl[:, :, :, self.parts_dict[part], 2])

                new_min0 = torch.min(img[:, :, :, self.parts_dict[part], 0])
                new_max0 = torch.max(img[:, :, :, self.parts_dict[part], 0])
                new_min1 = torch.min(img[:, :, :, self.parts_dict[part], 1])
                new_max1 = torch.max(img[:, :, :, self.parts_dict[part], 1])
                new_min2 = torch.min(img[:, :, :, self.parts_dict[part], 2])
                new_max2 = torch.max(img[:, :, :, self.parts_dict[part], 2])

                needed_0 = repl[:, :, :, self.parts_dict[part], 0]
                needed_1 = repl[:, :, :, self.parts_dict[part], 1]
                needed_2 = repl[:, :, :, self.parts_dict[part], 2]

                normalized0 = (needed_0 - old_min0) / (old_max0 - old_min0 + 1e-6)
                rescaled0 = normalized0 * (new_max0 - new_min0) + new_min0

                normalized1 = (needed_1 - old_min1) / (old_max1 - old_min1 + 1e-6)
                rescaled1 = normalized1 * (new_max1 - new_min1) + new_min1

                normalized2 = (needed_2 - old_min2) / (old_max2 - old_min2 + 1e-6)
                rescaled2 = normalized2 * (new_max2 - new_min2) + new_min2

                rescaled0 = torch.nan_to_num(rescaled0, 0.)
                rescaled1 = torch.nan_to_num(rescaled1, 0.)
                rescaled2 = torch.nan_to_num(rescaled2, 0.)

                img[:, :, :, self.parts_dict[part], 0] = rescaled0
                img[:, :, :, self.parts_dict[part], 1] = rescaled1
                img[:, :, :, self.parts_dict[part], 2] = rescaled2

            else:

                img[:, :, :, self.parts_dict[part], :] = repl[:, :, :, self.parts_dict[part], :]

        return img

    def get_one_item(self, index):
        yy = self.df[index]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)

        yy = torch.tensor(yy)

        yy = yy[:, self.interesting_idx, :]

        if self.train_mode:
            if random.random() < self.interpol_prob and yy.shape[0] > 10:
                yy = random_interpolate(yy)
            if random.random() < self.invert_prob:
                yy = flip_pose(yy)
            if random.random() < self.shift_prob:
                yy = self.shift_aug(yy)
            if random.random() < self.scale_prob:
                yy = scale_parts(yy)
            if random.random() < self.rotate_prob:
                yy = self.rotate_aug(yy)
            if random.random() < self.tree_rot_prob:
                yy[:, self.l_hand] = self.random_hand_rotate(yy[:, self.l_hand].numpy(),
                                                                degree=(-5, 5),
                                                                joint_prob=0.5)

                yy[:, self.r_hand] = self.random_hand_rotate(yy[:, self.r_hand].numpy(),
                                                                 degree=(-5, 5),
                                                                 joint_prob=0.5)

        if self.use_normalize:
            yy = self.norm(yy)
        yy = torch.nan_to_num(yy, 0.0)

        if random.random() < self.interp_nearest_random and yy.size(0) > self.new_size[0] and self.train_mode:
            indices = np.random.choice(yy.size(0), size=self.new_size[0], replace=False)
            indices.sort()
            yy = yy[indices]
            yy = yy[None, None, :]
        else:
            yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')  # this is base

        # yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')  # this is base

        return yy, self.Y[index]

    def __getitem__(self, i):

        if random.random() < self.mixup_prob and self.train_mode:
            base_img, base_y = self.get_one_item(i)

            mix_i = np.random.randint(len(self.Y))
            mix_img, mix_y = self.get_one_item(mix_i)

            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            img = base_img * mix_lambda + mix_img * (1 - mix_lambda)

            label = torch.zeros(250)
            label[base_y] = mix_lambda
            label[mix_y] = 1 - mix_lambda
        else:
            img, y = self.get_one_item(i)
            label = torch.zeros(250)
            label[y] = 1.

            if self.train_mode and random.random() < self.replace_prob:
                img = self.replace_aug(img, y)

        if random.random() < self.img_mask and self.train_mode:
            img = self.time_m(self.freq_m(img.squeeze(0).permute(0, 3, 1, 2)))[0]
        else:
            img = img[0, 0]
            img = img.permute(2, 0, 1)

        return img, label


class DatasetImageSmall112Mixup(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob
        self.scale_prob = cfg.scale_prob
        self.img_mask = cfg.img_masking
        self.shift_prob = cfg.shift_prob
        self.zero_prob = cfg.zero_prob
        self.rotate_prob = cfg.rotate_prob
        self.replace_prob = cfg.replace_prob
        self.interp_nearest_random = cfg.interp_nearest_random

        self.mixup_prob = cfg.mixup_prob

        self.train_mode = train_mode

        lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291][1:]  # [::2]#[1:]
        lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291][1:]  # [::2]#[1:]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308][1:]  # [::2]#[1:]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308][1:-2]  # [::2]#[1:]

        rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
        leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]

        LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
        LIPS = LIPS + rightEyeUpper0 + leftEyeUpper0
        pose = [489, 490, 492, 493, 494, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                507, 508, 509, 510, 511, 512]

        l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                  481, 482, 483, 484, 485, 486, 487, 488]
        r_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
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

        self.lips_ar = np.arange(0, 50)
        self.l_hand = np.arange(50, 71)
        self.pose_ar = np.arange(71, 91)
        self.r_hand = np.arange(91, 112)

        self.parts_dict = {'lips': self.lips_ar,
                           'rh': self.r_hand,
                           'lh': self.l_hand,
                           'pose': self.pose_ar}

    @staticmethod
    def flip_pose112(points):
        l_hand_indices = torch.tensor(np.arange(50, 71))
        pose_ind1_l, pose_ind1_r = torch.tensor(np.array([72, 73, 76]).astype(int)), torch.tensor(
            np.array([74, 75, 77]).astype(int))
        pose_ind2_l, pose_ind2_r = torch.tensor(np.array([78, 80, 82, 84, 86, 88]).astype(int)), torch.tensor(
            np.array([79, 81, 83, 85, 87, 89]).astype(int))
        r_hand_indices = torch.tensor(np.arange(91, 112))
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

    def rotate_aug(self, img):
        ax = np.random.choice(['x', 'y', 'z'])  # 1 or 2

        parts = np.random.choice(['lips', 'rh', 'lh', 'pose'], np.random.randint(1, 5), replace=False)

        for part in parts:
            img[:, self.parts_dict[part], :] = random_rotation_around_axis(img[:, self.parts_dict[part], :],
                                                                           axis=ax,
                                                                           min_angle=-10, max_angle=10)

        return img

    def replace_aug(self, img, lab):
        parts = np.random.choice(['lips', 'rh', 'lh', 'pose'], np.random.randint(1, 3), replace=False)

        for part in parts:
            repl_ind = np.random.choice(np.where(self.Y == lab)[0])
            repl, _ = self.get_one_item(repl_ind)

            img[:, :, :, self.parts_dict[part], :] = repl[:, :, :, self.parts_dict[part], :]

        return img

    def get_one_item(self, index):
        yy = self.df[index]

        if self.train_mode:
            if random.random() < self.aug_prob:
                yy = do_random_affine(yy, scale=(0.8, 1.3), shift=(-0.1, 0.1), degree=None)

        yy = torch.tensor(yy)

        yy = yy[:, self.interesting_idx, :]

        if self.train_mode:
            if random.random() < self.invert_prob:
                yy = self.flip_pose112(yy)
            if random.random() < self.shift_prob:
                yy = self.shift_aug(yy)
            if random.random() < self.scale_prob:
                yy = scale_parts(yy)
            if random.random() < self.rotate_prob:
                yy = self.rotate_aug(yy)

        yy = torch.nan_to_num(yy, 0.0)

        if random.random() < self.interp_nearest_random and yy.size(0) > self.new_size[0]:
            indices = np.random.choice(yy.size(0), size=self.new_size[0], replace=False)
            indices.sort()
            yy = yy[indices]
        else:
            yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')  # this is base

        return yy, self.Y[index]

    def __getitem__(self, i):

        if random.random() < self.mixup_prob and self.train_mode:
            base_img, base_y = self.get_one_item(i)

            mix_i = np.random.randint(len(self.Y))
            mix_img, mix_y = self.get_one_item(mix_i)

            mix_lambda = np.random.beta(10, 10)
            img = base_img * mix_lambda + mix_img * (1 - mix_lambda)

            label = torch.zeros(250)
            label[base_y] = mix_lambda
            label[mix_y] = 1 - mix_lambda
        else:
            img, y = self.get_one_item(i)
            label = torch.zeros(250)
            label[y] = 1.

            if self.train_mode and random.random() < self.replace_prob:
                img = self.replace_aug(img, y)

        if random.random() < self.img_mask and self.train_mode:
            img = self.time_m(self.freq_m(img.squeeze(0).permute(0, 3, 1, 2)))[0]
        else:
            img = img[0, 0]
            img = img.permute(2, 0, 1)

        return img, label


class DatasetImageSmall120Mixup(torch.utils.data.Dataset):
    def __init__(self, cfg, df, Y, train_mode=True):
        self.df = df
        self.Y = Y

        self.new_size = cfg.new_size

        self.aug_prob = cfg.aug_prob
        self.invert_prob = cfg.invert_prob
        self.scale_prob = cfg.scale_prob
        self.img_mask = cfg.img_masking
        self.shift_prob = cfg.shift_prob
        self.zero_prob = cfg.zero_prob
        self.rotate_prob = cfg.rotate_prob
        self.replace_prob = cfg.replace_prob

        self.mixup_prob = cfg.mixup_prob

        self.train_mode = train_mode

        lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
        leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]

        LIPS = list(set(lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner))
        LIPS = LIPS + rightEyeUpper0 + leftEyeUpper0
        pose = [489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                507, 508, 509, 510, 511, 512]

        l_hand = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
                  481, 482, 483, 484, 485, 486, 487, 488]
        r_hand = [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
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

        self.lips_ar = np.arange(0, 50)
        self.l_hand = np.arange(50, 71)
        self.pose_ar = np.arange(71, 91)
        self.r_hand = np.arange(91, 112)

        self.parts_dict = {'lips': self.lips_ar,
                           'rh': self.r_hand,
                           'lh': self.l_hand,
                           'pose': self.pose_ar}

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

    def rotate_aug(self, img):
        ax = np.random.choice(['x', 'y', 'z'])  # 1 or 2

        parts = np.random.choice(['lips', 'rh', 'lh', 'pose'], np.random.randint(1, 5), replace=False)

        for part in parts:
            img[:, self.parts_dict[part], :] = random_rotation_around_axis(img[:, self.parts_dict[part], :],
                                                                           axis=ax,
                                                                           min_angle=-10, max_angle=10)

        return img

    def replace_aug(self, img, lab):
        parts = np.random.choice(['lips', np.random.choice(['rh', 'lh']), 'pose'], np.random.randint(1, 3), replace=False)

        for part in parts:
            repl_ind = np.random.choice(np.where(self.Y == lab)[0])
            repl, _ = self.get_one_item(repl_ind)

            if part in ['rh', 'lh']:
                if (img[:, :, :, self.parts_dict[part], :] == 0).all():
                    part_in = 'rh' if part == 'lh' else 'lh'
                else:
                    part_in = part

                if (repl[:, :, :, self.parts_dict[part_in], :] == 0).all():
                    part_out = 'lh' if part_in == 'rh' else 'rh'
                    flipped_points = repl.clone()
                    x_max = flipped_points[:, :, 0].max()
                    flipped_points[:, :, :, :, 0] = x_max - flipped_points[:, :, :, :, 0]
                    flipped_points[:, :, :, self.parts_dict[part_in], :] = flipped_points[:, :, :, self.parts_dict[part_out], :]
                    repl = flipped_points

                img[:, :, :, self.parts_dict[part_in], :] = repl[:, :, :, self.parts_dict[part_in], :]
            else:

                img[:, :, :, self.parts_dict[part], :] = repl[:, :, :, self.parts_dict[part], :]

        return img

    def get_one_item(self, index):
        yy = self.df[index]

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
            if random.random() < self.rotate_prob:
                yy = self.rotate_aug(yy)

        yy = torch.nan_to_num(yy, 0.0)

        yy = F.interpolate(yy[None, None, :], size=self.new_size, mode='nearest')  # this is base

        return yy, self.Y[index]

    def __getitem__(self, i):

        if random.random() < self.mixup_prob and self.train_mode:
            base_img, base_y = self.get_one_item(i)

            mix_i = np.random.randint(len(self.Y))
            mix_img, mix_y = self.get_one_item(mix_i)

            mix_lambda = np.random.beta(10, 10)
            img = base_img * mix_lambda + mix_img * (1 - mix_lambda)

            label = torch.zeros(250)
            label[base_y] = mix_lambda
            label[mix_y] = 1 - mix_lambda
        else:
            img, y = self.get_one_item(i)
            label = torch.zeros(250)
            label[y] = 1.

            if self.train_mode and random.random() < self.replace_prob:
                img = self.replace_aug(img, y)

        if random.random() < self.img_mask and self.train_mode:
            img = self.time_m(self.freq_m(img.squeeze(0).permute(0, 3, 1, 2)))[0]
        else:
            img = img[0, 0]
            img = img.permute(2, 0, 1)

        return img, label


