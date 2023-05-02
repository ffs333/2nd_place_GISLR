import numpy as np
from scipy.interpolate import interp1d
from functools import partial

import numpy as np

NORM_REF = np.array([500, 501, 512, 513, 159,  386, 13,])

LIP_VERSION = "V2"
if LIP_VERSION == "V2":
    LIP = np.concatenate([
        [0],
        [13],
        [14],
        [17], # MID POINTS
        [267, 269, 270, 409, 291], # LEFT UPPER
        [312, 311, 310, 415, 308], # LEFT MID_UPPER
        [317, 402, 318, 324], # LEFT MID_LOWER
        [314, 405, 321, 375], # LEFT LOWER
        [37, 39, 40, 185, 61],
        [82, 81, 80, 191, 78],
        [87, 178, 88, 95],
        [84, 181, 91, 146]
    ])
    OLD_LIP = np.array([0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
    SIM_LIP = np.concatenate([
        [ 0], 
        [13], 
        [14], 
        [17], 
        [291], [267],  # LEFT
        [308], [312], 
               [317],
               [314],
        [61], [37],  # RIGHT
        [78], [82],  
              [87],  
              [84],  
    ])

    #   |----0------|
    #  /  |--13--|  \
    # 61  78    308 291
    # \   |--14--|  /
    # |------17----| 


    LIP_REIDX = {v: i for i, v in enumerate(LIP)}
    LIP_ANGLES = np.array([
        *[[0, 61, 17, 291, 0, 61][i:i + 3] for i in range(4)],
        *[[13, 78, 14, 308, 13, 78][i:i + 3] for i in range(4)],
    ])
    OLD_LIP_ANGLES = np.array([[LIP_REIDX[OLD_LIP[LIP_REIDX[_]]] for _ in _] for _ in LIP_ANGLES])
    NEW_LIP_ANGLES = np.array([[LIP_REIDX[_] for _ in _] for _ in LIP_ANGLES])
    LIP_ANGLES = OLD_LIP_ANGLES
    SIM_LIP = np.array([LIP_REIDX[_] for _ in SIM_LIP])
elif LIP_VERSION == "V3":
    # LIP = np.concatenate([
    #     [61], [185, 40,  39, 37],  [ 0], [267, 269, 270, 409], [291],
    #     [78], [191, 80,  81, 82],  [13], [312, 311, 310, 415], [308],
    #           [ 95, 88, 178, 87],  [14], [317, 402, 318, 324],
    #           [146, 91, 181, 84],  [17], [314, 405, 321, 375],
    # ])
    LIP = np.concatenate([
        [ 0], 
        [13], 
        [14], 
        [17], 
        [291], [267],  # LEFT
        [308], [312], 
               [317],
               [314],
        [61], [37],  # RIGHT
        [78], [82],  
              [87],  
              [84],  
    ])
    LIP_REIDX = {v: i for i, v in enumerate(LIP)}
    LIP_ANGLES = np.array([
        *[[0, 37, 61, 84, 17, 314, 291, 267, 0, 37][i:i + 3] for i in range(8)],
        *[[13, 82, 78, 87, 14, 317, 308, 312, 13, 82][i:i + 3] for i in range(8)],
    ])
    LIP_ANGLES = np.array([[LIP_REIDX[_] for _ in _] for _ in LIP_ANGLES])


HAND_ROUTES = [
    [0, *range(1, 5)], 
    [0, *range(5, 9)], 
    [0, *range(9, 13)], 
    [0, *range(13, 17)], 
    [0, *range(17, 21)],
    # [4,8,12,16,20],
    # [3,7,11,15,19],
    # [2,6,10,14,18],
    # [1,5,9,13,17],
]
HAND_ANGLES = np.array(sum([[route[i:i + 3] for i in range(len(route) - 2)] for route in HAND_ROUTES], []))
HAND_EDGES = np.array(sum([[route[i:i + 2] for i in range(len(route) - 1)] for route in HAND_ROUTES], []))
HAND_TREES = sum([[np.array(route[i:]) for i in range(len(route) - 1)] for route in HAND_ROUTES], [])


BODY = np.array([16, 14, 12, 11, 13, 15])
BODY_REIDX = {v: i for i, v in enumerate(BODY)}
BODY = BODY + 468 + 21
BODY_ANGLES = np.array([
    [16, 14, 12],
    [14, 12, 11],
    [11, 13, 15],
    [13, 11, 12],
])
BODY_ANGLES = np.array([[BODY_REIDX[_] for _ in _] for _ in BODY_ANGLES])
BODY_EDGES = np.array([
    [16, 14], [14, 12],
    [12, 11], 
    [11, 13], [13, 15],
])
BODY_EDGES = np.array([[BODY_REIDX[_] for _ in _] for _ in BODY_EDGES])

LEFT_EYE = np.concatenate([
    [263, 466, 388, 387, 386, 385, 384, 398, 362], # LEFT UPPER
    [249, 390, 373, 374, 380, 381, 382] # LEFT LOWER
])
RIGHT_EYE = np.concatenate([
    [33, 246, 161, 160, 159, 158, 157, 173, 133], # RIGHT UPPER (including two midpoint)
    [7, 163, 144, 145, 153, 154, 155], # RIGHT LOWER
])
EYE = np.concatenate([LEFT_EYE, RIGHT_EYE])

def random_affine(pos, scale  = (0.8, 1.5), shift  = (-0.1, 0.1), degree = (-15, 15), p = 0.5):
    if np.random.rand() < p:
        if scale is not None:
            scale = np.random.uniform(*scale)
            pos = scale * pos
        if shift is not None:
            shift = np.random.uniform(*shift, (1,1,pos.shape[-1]))
            pos = pos + shift
        if degree is not None:
            degree = np.random.uniform(*degree)
            radian = degree / 180 * np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([[c, -s], [s, c]]).T
            pos[...,:2] = pos[...,:2] @ rotate
    return pos


def random_interpolate(pos, scale = (0.8, 1.5), shift = (-0.5, 0.5), p = 0.5):
    if np.random.rand() < p:
        scale = np.random.uniform(*scale)
        shift = np.random.uniform(*shift)
        orig_time = np.arange(len(pos))
        inte = interp1d(orig_time, pos, axis = 0, fill_value = "extrapolate")
        new_time = np.linspace(0 + shift, len(pos) - 1 + shift, int(round(len(pos) * scale)), endpoint = True)
        pos = inte(new_time).astype(np.float32)
    return pos


def random_maskout(pos, mask_prob = 0.15, p = 0.5):
    if np.random.rand() < p:
        pos[np.random.rand(len(pos)) < mask_prob] = 0.0
    return pos


def rotate_points(pos, center, alpha):
    radian = alpha / 180 * np.pi
    rotation_matrix = np.array([[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]])
    translated_points = (pos - center).reshape(-1, 2)
    rotated_points = np.dot(rotation_matrix, translated_points.T).T.reshape(*pos.shape)
    rotated_pos = rotated_points + center
    return rotated_pos


def random_hand_rotate(pos, degree=(-4, 4), joint_prob=0.15, p=0.5):
    if np.random.rand() < p:
        for tree in HAND_TREES:
            if np.random.rand() < joint_prob:
                alpha = np.random.uniform(*degree)
                center = pos[:,tree[0:1],:2]
                pos[:,tree[1:],:2] = rotate_points(pos[:,tree[1:],:2], center, alpha)
    return pos

def random_hand_limb_scale(pos, scale = (-0.05, 0.05), joint_prob = 0.15, p = 0.5):
    if np.random.rand() < p:
        for tree in HAND_TREES:
            if np.random.rand() < joint_prob:
                percent = np.random.uniform(*scale)
                target = pos[:,tree[0:1],:2]
                source = pos[:,tree[1:2],:2]
                pos[:,tree[1:],:2] += (target - source) * percent
    return pos


def random_lip_scale(pos, scale = (0.9, 1.2), p = 0.5):
    if np.random.rand() < p:
        scale = np.random.uniform(*scale, size = (1, 1, 3))
        pos = scale * pos
    return pos


def random_hand_shift(pos, shift = (-0.01, 0.01), p = 0.5):
    if np.random.rand() < p:
        shift = np.random.uniform(*shift, (1,1,2))
        pos[...,:2] = pos[...,:2] + shift
    return pos


def random_noise(pos, sigma, p = 0.5):
    if np.random.rand() < p:
        pos = pos + np.random.normal(0, sigma, size = pos.shape)
    return pos


def random_flip_hand(pos, ls, le, rs, re, p = 0.5):
    if np.random.rand() < p:
        center = pos[:,0:1,0]
        frhand, flhand = pos[:,ls:le], pos[:,rs:re]
        flhand[...,0], frhand[...,0] = 2 * center - flhand[...,0], 2 * center - frhand[...,0]
        pos[:,ls:le], pos[:,rs:re] = flhand, frhand
    return pos


def random_lip_drop_out(lip, p = 0.2):
    mask = np.random.rand(lip.shape[1]) < p
    lip[:,mask] = np.nan
    return lip


def random_flip_lip(lip, p = 0.5):
    if np.random.rand() < p:
        center = lip[:,0:1,0]# .mean(1, keepdims = True)
        lip[:,:,0] = 2 * center - lip[:,:,0]
        # pos[:,4:22,0], pos[:,22:40,0] = 2 * center - pos[:,22:40,0], 2 * center - pos[:,4:22,0]
        lip[:,4:22], lip[:,22:40] = lip[:,22:40], lip[:,4:22]
    return lip


def flip_hand(lip, hand):
    center = lip[:,0:1,0].mean(1, keepdims = True)
    hand[...,0] = 2 * center - hand[...,0]
    return hand

random_hand_op_h2 = partial(random_hand_rotate, joint_prob = 0.2, p = 0.8)
random_hand_op_h4 = partial(random_hand_rotate, joint_prob = 0.4, p = 0.9)
random_hand_op_h5 = partial(random_hand_rotate, joint_prob = 0.8, p = 1.0)
random_hand_op_h4_1 = lambda hand: random_hand_limb_scale(random_hand_rotate(hand, joint_prob = 0.4, p = 0.9), joint_prob = 0.4, p = 0.9)
# random_hand_op_h4_2 = partial(random_hand_rotate_byframe, joint_prob = 0.4, p = 0.9)

random_lip_op_l1 = partial(random_flip_lip, p = 0.5)
random_lip_op_l2 = partial(random_lip_drop_out, p = 0.2)
aug2 = lambda pos: random_interpolate(random_affine(pos, p = 0.8), p = 0.3)

# a2: p=0.8 affine, p=0.3 interpolate
# a3: remove hand-nan slice
# h2: use larger hand rotate prob: joint_prob=0.2, p=0.8
# h4: use larger hand rotate prob: joint_prob=0.4, p=0.9
# h4.1: rotate and scale hand limbs: joint_prob=0.8, p=1
# h5: use larger hand rotate prob: joint_prob=0.8, p=1
# h3: use lip midpoint mean-x as center
# l1: random flip lip
# l2: random lip drop out p=0.2
# l3: random lip drop out p=0.1
# lp: pre restrict max-length (after augmentation)
# f: -x flip
# fx: -x flip + 3d distance + 2d angle of hand
# M1: L4D256D128
# M1m: L4D256D128 + all token mean logits
# M2: D256D192
# M3: D256D512
# M3.1: D300D512
# M3.1: D224D512
# M3.2: D256D256
# cpM3: D256D512 + concat position embedding (256 = 224 + 32)
# npM3: D256D512 + no position embedding
# M4: D192D384
# M5: D256D128AttnD128 (Squeeze Attn)
# L1: LSTM follow M1
# d2d2at: add lip triu distance
# d2d2a2t: add lip angles
# d2d2a3t: debug lip angle index
# d2d2a4t: change lip npy data to new LIP, use OLD_LIP_ANGLE
# d2d2a5t: change lip npy data to new LIP, use NEW_LIP_ANGLE
# d2d2a4tz: no z
# nd2d2a4t: zyx-wise normalize
# n2d2d2a4t: simplest zyx-wise normalize (https://www.kaggle.com/competitions/asl-signs/discussion/391265#2206111)


# N1: a2h4f_d2d2a4tlsp_M3E70-45_sm50_cpe1-4
# nolipz: remove lip'z in pos-dpos-rdpos (and normalize)