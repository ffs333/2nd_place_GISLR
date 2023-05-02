import os
import json
import random

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score


ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed=42):
    """
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # was commented
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def my_interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


def load_data(_cfg):

    if 'tiny' in _cfg.model:
        print(f'Load data for tiny model')
        X_base = torch.load(_cfg.base_path + 'generated_data/X_stack.pt')
        X_invert = torch.load(_cfg.base_path + 'generated_data/X_stack_invert.pt')
        XM_base = np.load(_cfg.base_path + 'generated_data/feature_data_8x.npy')
        XM_invert = np.load(_cfg.base_path + 'generated_data/feature_data_8x_invert.npy')
        labels = np.load(_cfg.base_path + 'generated_data/Y.npy')
        return {'X': X_base, 'X_inv': X_invert,
                'XM': XM_base, 'XM_inv': XM_invert, 'Y': labels}
    elif 'V2' in _cfg.dataset:
        X = torch.load(_cfg.base_path + 'gen_data2/X.pt')
        X_aug = torch.load(_cfg.base_path + 'gen_data2/X_aug.pt')
        X_invert = torch.load(_cfg.base_path + 'gen_data2/X_invert.pt')
        X_aug_invert = torch.load(_cfg.base_path + 'gen_data2/X_aug_invert.pt')

        X_tf = np.load(_cfg.base_path + 'gen_data2/X_tf.npy')
        X_tf_aug = np.load(_cfg.base_path + 'gen_data2/X_tf_aug.npy')
        X_tf_invert = np.load(_cfg.base_path + 'gen_data2/X_tf_invert.npy')
        X_tf_aug_invert = np.load(_cfg.base_path + 'gen_data2/X_tf_aug_invert.npy')

        labels = np.load(_cfg.base_path + 'gen_data2/Y.npy')

        return {'X': X, 'X_inv': X_invert, 'X_aug': X_aug, 'X_aug_invert': X_aug_invert,
                'X_tf': X_tf, 'X_tf_aug': X_tf_aug, 'X_tf_invert': X_tf_invert, 'X_tf_aug_invert': X_tf_aug_invert,
                'Y': labels}

    else:
        X_base = torch.load(_cfg.base_path + 'generated_data/X.pt')
        X_invert = torch.load(_cfg.base_path + 'generated_data/X_invert.pt')
        XM_base = np.load(_cfg.base_path + 'generated_data/feature_data.npy')
        XM_invert = np.load(_cfg.base_path + 'generated_data/feature_data_invert.npy')
        labels = np.load(_cfg.base_path + 'generated_data/Y.npy')

        return {'X': X_base, 'X_inv': X_invert,
                'XM': XM_base, 'XM_inv': XM_invert, 'Y': labels}


def get_scores(labs, prds, cfg):
    acc = accuracy_score(labs, prds.argmax(1))

    topk_score = torch.topk(prds, cfg.k).indices

    cnt = 0
    for i in range(len(labs)):
        if labs[i] in topk_score[i]:
            cnt += 1
    return acc, cnt / len(labs)


def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic

