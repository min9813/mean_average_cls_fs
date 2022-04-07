import numpy as np
import os
import glob
import argparse
import backbone
import metaopt_models.backbone_metaopt as backbone_metaopt

import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import math
import gzip
import numpy
import yaml
from easydict import EasyDict as edict


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def load_cifar_pickle(pickle_path):
    with open(pickle_path, "rb") as pkl:
        data = pickle.load(pkl, encoding="bytes")

    use_keys = list(data.keys())
    for key in use_keys:
        if isinstance(key, str):
            continue
        data[key.decode("ascii")] = data[key]
        data.pop(key)
        
    """
    data: {
        `data`: (num_image, 32, 32, 3),
        `label`: (num_image,)
    }
    """
        
    return data


def load_json(path, print_path=False):
    if print_path:
        print("load json from", path)

    path = str(path)

    with open(path, "r") as f:
        data = json.load(f)

    return data


def load_pickle(path, print_path=False, is_gzip=False):
    if print_path:
        print("load pickle from", path)

    path = str(path)

    if is_gzip:
        with gzip.open(path, "rb") as pkl:
            data = pickle.load(pkl)

    else:
        with open(path, "rb") as pkl:
            data = pickle.load(pkl)

    return data


def save_json(data, path, print_path=False, is_gzip=False):
    if print_path:
        print("save json to", path)

    path = str(path)
    with open(path, "w") as f:
        json.dump(data, f, cls=MyEncoder)


def save_pickle(data, path, print_path=False, is_gzip=False):
    if print_path:
        print("save pickle to", path)

    path = str(path)

    if is_gzip:
        with gzip.open(path, "wb") as pkl:
            pickle.dump(data, pkl)

    else:
        with open(path, "wb") as pkl:
            pickle.dump(data, pkl)

    return data


def load_yaml(filename):
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    return yaml_cfg


def load_h5py_file(path):
    import h5py
    with h5py.File(path, "r") as f:
        feats = f["all_feats"][...]
        labels = f["all_labels"][...]
        counts = f["count"][...]

    data = {
        "all_feats": feats,
        "all_labels": labels,
        "count": counts
    }

    return data


def append_json_to_file(data, path_file):
    os.makedirs(str(pathlib.Path(path_file).parent), exist_ok=True)
    if os.path.exists(path_file) is False:
        with open(path_file, "w") as f:
            f.write('')
    with open(path_file, 'ab+') as f:              # ファイルを開く
        f.seek(0,2)                                # ファイルの末尾（2）に移動（フォフセット0）  
        if f.tell() == 0 :                         # ファイルが空かチェック
            f.write(json.dumps([data], cls=MyEncoder).encode())   # 空の場合は JSON 配列を書き込む
        else :
            f.seek(-1,2)                           # ファイルの末尾（2）から -1 文字移動
            f.truncate()                           # 最後の文字を削除し、JSON 配列を開ける（]の削除）
            f.write(' , '.encode())                # 配列のセパレーターを書き込む
            f.write(json.dumps(data, cls=MyEncoder).encode())     # 辞書を JSON 形式でダンプ書き込み
            f.write(']'.encode()) 


def load_area_file(path):
    import cv2
    import numpy as np
    img = cv2.imread(path)
    img = np.max(img, axis=2)

    return img



model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            ResNet12=backbone.ResNet12,
            ResNet12Metaopt=backbone_metaopt.resnet12
            )

def str2bool(var):
    return "t" in var.lower()


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char/cifarfs/fc100/tieredImagenet')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_episode'      , default=100, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--support_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--debug'   , action='store_true',  help='debug mode') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--support_aug_num', default=5, type=int)
    parser.add_argument('--test_as_protonet', action="store_true")
    parser.add_argument('--normalize_method', default="no", choices=[
        "no", "l2", "maha", "maha-diag_cov", 
        "support_to_mean_of_norm", "minimize_variance_of_norm_by_trace", 
        "maha-diag_cov_all", "est_test", "est_beforehand", "est_beforehand_l2",
        "est_beforehand_z_score", "z_score", "force_z_score-est_beforehand", "force_z_score-est_beforehand_after_z_score",
        "lda_test", "l2_lda_test_after_l2", "l2_est_test_after_l2", "lda_test_after_l2",
        "l2_lda_test", "l2_est_test", "est_beforehand_l2_est_test", "est_beforehand_l2_est_test_after_l2",
        "force_l2-maha-diag_cov", "force_l2-est_test", "force_l2-lda_test", "force_l2-est_beforehand",
        "est_test_after_l2", "force_l2-est_beforehand_l2", "force_l2-est_test_after_l2",
        "force_l2-est_beforehand_after_l2"
        ])
    parser.add_argument('--del_last_relu', action="store_true")
    parser.add_argument('--loss_type', default="no")
    parser.add_argument("--get_latest_file", action="store_true")
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--norm_factor", type=float, default=1.)
    parser.add_argument('--force_normalize++', default="f", type=str2bool)
    parser.add_argument("--normalize_query", type=str2bool, default="t")
    parser.add_argument("--normalize_vector", type=str, default="before_and_mean")
    parser.add_argument("--subtract_mean", type=str2bool, default="f")
    parser.add_argument("--adjust_to_mean_norm", type=str2bool, default="f")
    parser.add_argument('--add_final_layer', type=str2bool, default="f")
    parser.add_argument('--subtract_mean_method', default="", choices=[
        "mean", "cov_normalize", "diag_cov_normalize", "equal_angle_mean_start",
        "equal_angle_zero_start"
        ])

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper
    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--test_lr'  , default=0.01, action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--test_epoch'  , default=100, action='store_true', help='further adaptation in test time or not')
    else:
       raise ValueError('Unknown script')
        

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
