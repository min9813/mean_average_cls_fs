from numpy.core.fromnumeric import mean
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
from methods import meta_test_preprocess
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
    # print(z_all.shape)
    # sfa

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
    if scores.shape[1] > n_way:
        n_prototype = scores.shape[1] // n_way
        assert n_way * n_prototype == scores.shape[1], (scores.shape, n_way)
        pred_label = np.repeat(np.arange(n_way), n_prototype)
        # assert scores.shape[1] == n_way * n_support, (n_way, n_support, scores.shape)
        # pred = pred // n_support
        pred = pred_label[pred]
    y = np.repeat(range(n_way), n_query)
    # print(pred)
    # print(y)
    # sfda
    acc = np.mean(pred == y)*100
    return acc


if __name__ == '__main__':
    params = parse_args('test')

    # if "force_l2" in params

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
            lr=params.test_lr,
            epoch=params.test_epoch,
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
            lr=params.test_lr,
            epoch=params.test_epoch,
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

    if params.add_final_layer:
        final_layer_name = "-add_final_layer"

    else:
        final_layer_name = ""
    params.checkpoint_dir += final_layer_name

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
        # print("model file:", modelfile)
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


        # print("novel file:", novel_file)
        # print("checkpoint dir:", checkpoint_dir)

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

        print("novel file:", novel_file)
        if "best" not in novel_file and params.save_iter == -1 and "best" not in modelfile:
            novel_file = novel_file.replace(".hdf5", "_best_model.hdf5")

        if os.path.exists(novel_file) is False:
            print("model file:", modelfile)
            file_stem = pathlib.Path(modelfile).stem
            novel_file = novel_file.replace(
                ".hdf5", "_{}.hdf5".format(file_stem))

        if os.path.exists(novel_file) is False and params.add_final_layer:
            novel_file = novel_file.replace(
                f"{split}", "{}-add_final_layer".format(split))

        print("load feature from", novel_file)
        cl_data_file = feat_loader.init_loader(novel_file)

        if "force_l2" in params.normalize_method:
            cl_data_file = meta_test_preprocess.l2_normalize_all(
                cl_data_file=cl_data_file
            )

        if "force_z_score" in params.normalize_method:
            cl_data_file = meta_test_preprocess.zscore_normalize_all(
                cl_data_file=cl_data_file
            )

        # print(novel_file)
        # sdfa
        if params.subtract_mean and "est_beforehand" not in params.normalize_method:
        # if params.subtract_mean:
            base_file = novel_file.replace("novel", "base")
            cl_data_file_base = feat_loader.init_loader(base_file)
            mean_feature = meta_test_preprocess.calc_center_point(
                feature_info=cl_data_file_base,
                method=params.subtract_mean_method
            )
            mean_feature = mean_feature[0]
            # mean_feature = None
            # print(mean_feature.shape)
            # sfad
            # print(mean_feature_norm)
        else:
            mean_feature = None
            # sdfa
        if params.adjust_to_mean_norm:
            cl_data_file = meta_test_preprocess.adjust_to_class_mean_norm(
                cl_data_file=cl_data_file
            )

        if params.normalize_method == "minimize_variance_of_norm_by_trace":
            base_file = novel_file.replace("novel", "base")
            cl_data_file_base = feat_loader.init_loader(base_file)

            cl_data_file = meta_test_preprocess.minimize_variance_beforehand(
                base_features=cl_data_file_base,
                novel_features=cl_data_file
            )

        elif "est_beforehand" in params.normalize_method:
            base_file = novel_file.replace("novel", "base")
            print("load base from", base_file)
            cl_data_file_base = feat_loader.init_loader(base_file)

            if "force_l2" in params.normalize_method:
                cl_data_file_base = meta_test_preprocess.l2_normalize_all(
                    cl_data_file=cl_data_file_base
                )
                
            if "force_z_score" in params.normalize_method:
                cl_data_file_base = meta_test_preprocess.zscore_normalize_all(
                    cl_data_file=cl_data_file_base
                )
            # if "l2" in params.normalize_method:
            #     cl_data_file_base = meta_test_preprocess.l2_normalize_all(
            #         cl_data_file=cl_data_file_base
            #     )

            cl_data_file, cl_data_file_base = meta_test_preprocess.est_beforehand(
                base_features=cl_data_file_base,
                novel_features=cl_data_file
            )

            if params.subtract_mean:
                base_file = novel_file.replace("novel", "base")
            #     # cl_data_file_base = feat_loader.init_loader(base_file)
                mean_feature = meta_test_preprocess.calc_center_point(
                    feature_info=cl_data_file_base,
                    method=params.subtract_mean_method
                )
                mean_feature = mean_feature[0]
                # print(mean_feature.shape)
                # sfda

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

    save_record_dir = f"./record/{params.dataset}"
    os.makedirs(save_record_dir, exist_ok=True)
    with open(f'{save_record_dir}/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        aug_str = '-aug' if params.train_aug else ''
        if params.support_aug:
            aug_str += "-support_aug_{}".format(params.support_aug_num)

        if params.test_as_protonet:
            aug_str += "-test_as_protonet"

        aug_str += "-norm_factor={}".format(params.norm_factor)

        aug_str += "-loss:{}".format(params.loss_type)

        aug_str += "-subtract_mean:{}".format(params.subtract_mean)
        if params.subtract_mean:
            aug_str += "-mean_method:{}".format(params.subtract_mean_method)

        aug_str += "-adjust_mean_norm:{}".format(params.adjust_to_mean_norm)
        # if params.normalize_feature:
        aug_str += "-normalize:{}".format(params.normalize_method)
        aug_str += "-normvec:{}".format(params.normalize_vector)
        aug_str += "-normquery:{}".format(params.normalize_query)

        aug_str += "-dim:{}".format(params.output_dim)

        if params.add_final_layer:
            aug_str += "-add_final_layer"

        else:
            aug_str += "-normal_final_layer"
        aug_str += '-adapted' if params.adaptation else ''

        hyperparameter_str = f"-test_lr:{params.test_lr}-test_epoch:{params.test_epoch}"
        aug_str += hyperparameter_str

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
