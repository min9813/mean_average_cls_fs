import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
import pathlib

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from tqdm import tqdm
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False, mean_feature=None):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]])
                      for i in range(n_support+n_query)])     # stack each batch

    z_all = np.array(z_all)

    if mean_feature is not None:
        z_all = z_all - mean_feature[None, None, :]

    z_all = torch.from_numpy(z_all)

    # z_all = z_all / torch.sqrt(torch.sum(z_all*z_all, dim=2, keepdim=True))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    # print(pred)
    # print(y)
    # sfda
    acc = np.mean(pred == y)*100
    return acc


if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method == 'baseline' and not params.test_as_protonet:
        if params.loss_type == "no":
            loss_type = "softmax"

        else:
            loss_type = params.loss_type
        model = BaselineFinetune(
            model_dict[params.model],
            normalize_method=params.normalize_method,
            loss_type=loss_type,
            del_last_relu=params.del_last_relu,
            output_dim=params.output_dim,
            subtract_mean=params.subtract_mean,
            **few_shot_params)
    elif params.method == 'baseline++' and not params.test_as_protonet:
        if params.loss_type == "no":
            loss_type = "dist"

        else:
            loss_type = params.loss_type

        model = BaselineFinetune(
            model_dict[params.model], 
            normalize_method=params.normalize_method, 
            subtract_mean=params.subtract_mean,
            loss_type=loss_type, 
            **few_shot_params)
    elif params.method == 'protonet' or params.test_as_protonet:
        model = ProtoNet(model_dict[params.model],
                         normalize_method=params.normalize_method,
                         norm_factor=params.norm_factor,
                         normalize_vector=params.normalize_vector,
                         normalize_query=params.normalize_query,
                         subtract_mean=params.subtract_mean,
                         output_dim=params.output_dim,
                         del_last_relu=params.del_last_relu,
                         debug=params.debug, **few_shot_params)
    elif params.method == 'matchingnet':
        model = MatchingNet(model_dict[params.model], **few_shot_params)
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            def feature_model(): return model_dict[params.model](flatten=False)
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model = RelationNet(
            feature_model, loss_type=loss_type, **few_shot_params)
    elif params.method in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[params.model], approx=(
            params.method == 'maml_approx'), **few_shot_params)
        # maml use different parameter in omniglot
        if params.dataset in ['omniglot', 'cross_char']:
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    if params.del_last_relu:
        del_relu_name = "del-last-relu"
        # params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' % (
        # configs.save_dir, params.dataset, params.model, del_relu_name, params.method)

    else:
        del_relu_name = "has-relu"
        # params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        # configs.save_dir, params.dataset, params.model, params.method)

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s_%s' % (
        configs.save_dir, params.dataset, params.model, del_relu_name, params.method, str(params.output_dim))

    checkpoint_dir = params.checkpoint_dir

    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] or params.support_aug:
        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            # print(tmp.keys())
            # print(tmp["state"].keys())
            state = tmp['state']
            if params.method in ["baseline", "baseline++"]:
                state_keys = list(state.keys())
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                        newkey = key.replace("feature.", "")
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)

                model.feature.load_state_dict(state)

            else:
                model.load_state_dict(state)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split
    # maml do not support testing with feature
    if params.method in ['maml', 'maml_approx'] or params.support_aug:
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 84

        datamgr = SetDataManager(
            image_size, n_eposide=iter_num, n_query=15, **few_shot_params)

        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                loadfile = configs.data_dir['CUB'] + split + '.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
            else:
                loadfile = configs.data_dir['emnist'] + split + '.json'
        else:
            loadfile = configs.data_dir[params.dataset] + split + '.json'

        if params.support_aug:
            novel_loader = datamgr.get_data_loader(
                loadfile, aug=False, is_support_aug=True, aug_num=params.support_aug_num)

        else:
            novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        if params.adaptation:
            # We perform adaptation on MAML simply by updating more times.
            model.task_update_num = 100
        model.eval()
        acc_mean, acc_std = model.test_loop(
            novel_loader, return_std=True, params=params)

    else:
        # defaut split = novel, but you can also test base or val classes
        novel_file = os.path.join(checkpoint_dir.replace(
            "checkpoints", "features"), split_str + ".hdf5")

        if params.del_last_relu:
            novel_file = novel_file.replace(".hdf5", "_del_last_relu.hdf5")

        if params.save_iter != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    #    elif params.method in ['baseline', 'baseline++'] :
    #        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
        else:
            if params.get_latest_file:
                modelfile = get_resume_file(checkpoint_dir)
                file_stem = pathlib.Path(modelfile).stem
                novel_file = novel_file.replace(
                    ".hdf5", "_{}.hdf5".format(file_stem))
            else:
                modelfile = get_best_file(checkpoint_dir)

        if os.path.exists(novel_file) is False:
            file_stem = pathlib.Path(modelfile).stem
            novel_file = novel_file.replace(
                ".hdf5", "_{}.hdf5".format(file_stem))

        cl_data_file = feat_loader.init_loader(novel_file)

        # print(novel_file)
        # sdfa
        if params.subtract_mean:
            all_feature_list = []
            for label, feature_list in cl_data_file.items():
                feature_list = np.array(feature_list)
                all_feature_list.append(feature_list)

            all_feature_list = np.array(all_feature_list)
            n_class, n_support, dim = all_feature_list.shape

            all_feature_list = all_feature_list.reshape(n_class*n_support, dim)
            mean_feature = np.mean(all_feature_list, axis=0)
            mean_feature_norm = np.sqrt(np.sum(mean_feature*mean_feature))
            # print(mean_feature_norm)
            # sdfa

        else:
            mean_feature = None

        for i in tqdm(range(iter_num)):
            acc = feature_evaluation(
                cl_data_file, model, n_query=15,
                mean_feature=mean_feature,
                adaptation=params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))
    with open('./record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        if params.support_aug:
            aug_str += "-support_aug_{}".format(params.support_aug_num)

        if params.test_as_protonet:
            aug_str += "-test_as_protonet"

        aug_str += "-norm_factor={}".format(params.norm_factor)

        aug_str += "-loss:{}".format(params.loss_type)

        aug_str += "-subtract_mean:{}".format(params.subtract_mean)
        # if params.normalize_feature:
        aug_str += "-normalize:{}".format(params.normalize_method)
        aug_str += "-normvec:{}".format(params.normalize_vector)
        aug_str += "-normquery:{}".format(params.normalize_query)

        aug_str += "-dim:{}".format(params.output_dim)

        aug_str += '-adapted' if params.adaptation else ''
        if params.method in ['baseline', 'baseline++']:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' % (
                params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way)
        else:
            exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' % (
                params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' % (
            iter_num, acc_mean, 1.96 * acc_std/np.sqrt(iter_num))
        f.write('Time: %s, Setting: %s, Acc: %s \n' %
                (timestamp, exp_setting, acc_str))
