import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import logging
import logger_config

import configs
import backbone
import metaopt_models
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.metaopt import MetaOpt
from io_utils import model_dict, parse_args, get_resume_file


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params, logger, trn_logger):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    if params.dataset == "tieredImagenet" and params.method in ['baseline', 'baseline++']:
        early_stop_epoch = 10
        eval_freq = 1

    elif params.method == "protonet" and params.dataset in ("fc100", "cifarfs"):
        early_stop_epoch = 100
        eval_freq = 1

    else:
        eval_freq = 5
        early_stop_epoch = 3400

    train_st = time.time()
    iter_st = train_st
    best_epoch = None
    for epoch in range(start_epoch, stop_epoch+1):
        model.train()
        # model are called by reference, no need to return
        model.train_loop(epoch, base_loader,  optimizer, trn_logger=trn_logger, params=params)
        model.eval()
        # sfa

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if epoch % eval_freq == 0 or epoch == stop_epoch:
            acc = model.test_loop(val_loader, is_aug=params.support_aug, params=params)
            if isinstance(acc, tuple):
                acc, acc_mean, acc_std = acc

            acc_mean = np.mean(acc)
            acc_err = np.std(acc) / np.sqrt(len(acc))        # print(acc)
        else:
            acc_mean = -1

        iter_time = time.time() - iter_st
        iter_st = time.time()
        total_time = time.time() - train_st

        msg = "Epoch [{} / {}] acc:{:.2f} +- {:.2f} total time:{:.1f}s mean_epoch_time:{:.1f}s".format(epoch, stop_epoch, acc_mean, acc_err, total_time, iter_time)
        logger.info(msg)
        if acc_mean > max_acc:  # for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            logger.info("best model in {}!  acc={:.2f} > {:.2f} save...".format(epoch, acc_mean, max_acc))
            max_acc = acc_mean
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            
            torch.save(
                {'epoch': epoch,
                 'state': model.state_dict(),
                 "acc": acc_mean,
                 "acc_std": acc_err},
                outfile)
            best_epoch = epoch
            is_break = False

        elif best_epoch is not None:
            if epoch - best_epoch >= early_stop_epoch:
                logger.info("not improve from epoch {} (patience={})".format(best_epoch, early_stop_epoch))
                is_break = True

            else:
                is_break = False

        else:
            is_break = False

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch-1) or is_break:
            outfile = os.path.join(params.checkpoint_dir,
                                   '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if model.debug:
            if epoch >= 2:
                break

        if is_break:
            break

    return model


dataset2num_classes = {
    "tieredImagenet": 351,
    "cifarfs": 64,
    "fc100": 60,
    "miniImagenet": 200
}


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        if params.dataset == "fc100":
            base_file = configs.data_dir[params.dataset] + 'FC100_train.pickle'
            val_file = configs.data_dir[params.dataset] + 'FC100_val.pickle'
            # params.num_classes = 60

        else:
            base_file = configs.data_dir[params.dataset] + 'CIFAR_FS_train.pickle'
            val_file = configs.data_dir[params.dataset] + 'CIFAR_FS_val.pickle'

        params.num_classes = dataset2num_classes[params.dataset]

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 84

    if params.dataset in ["cifarfs", "fc100"]:
        image_size = 32

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.dataset == "tieredImagenet":
        batch_size = 64

    else:
        batch_size = 16

    # print("batch size:", batch_size)
    # sdfa

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
                params.stop_epoch = 200
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=batch_size)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(
                model_dict[params.model], params.num_classes, debug=params.debug, del_last_relu=params.del_last_relu, output_dim=params.output_dim, add_final_layer=params.add_final_layer)
        elif params.method == 'baseline++':
            model = BaselineTrain(
                model_dict[params.model], params.num_classes, loss_type='dist', debug=params.debug, del_last_relu=params.del_last_relu, output_dim=params.output_dim, add_final_layer=params.add_final_layer)

    elif params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx', "metaopt_svm"]:
        # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        n_query = max(1, int(16 * params.test_n_way/params.train_n_way))

        train_few_shot_params = dict(
            n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(
            image_size, n_query=n_query, n_episode=params.n_episode, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)

        test_few_shot_params = dict(
            n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(
            image_size, n_query=n_query, n_episode=params.n_episode, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], debug=params.debug, 
                output_dim=params.output_dim,
                del_last_relu=params.del_last_relu,
                add_final_layer=params.add_final_layer,
                **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(
                model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                def feature_model(): return model_dict[params.model](
                    flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(
                feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(
                params.method == 'maml_approx'), **train_few_shot_params)
            # maml use different parameter in omniglot
            if params.dataset in ['omniglot', 'cross_char']:
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
        
        elif params.method in ["metaopt_svm"]:
            model = MetaOpt(model_dict[params.model], debug=params.debug, 
                output_dim=params.output_dim,
                del_last_relu=params.del_last_relu,
                add_final_layer=params.add_final_layer,
                **train_few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    if params.del_last_relu:
        del_relu_name = "del-last-relu"

    else:
        del_relu_name = "has-relu"

    if params.add_final_layer:
        final_layer_name = "-add_final_layer"

    else:
        final_layer_name = ""

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s_%s' % (
        configs.save_dir, params.dataset, params.model, del_relu_name, params.method, str(params.output_dim))

    params.checkpoint_dir += final_layer_name

    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if params.support_aug:
        params.checkpoint_dir += '_suppport-aug'


    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (
            params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    msglogger = logger_config.config_pylogger(
        './config/logging.conf', output_dir=params.checkpoint_dir)
    trn_logger = logging.getLogger().getChild('train')
    val_logger = logging.getLogger().getChild('valid')

    model.trn_logger = trn_logger
    model.val_logger = val_logger

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        # maml use multiple tasks in one update
        stop_epoch = params.stop_epoch * model.n_task

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
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
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader,  model,
                  optimization, start_epoch, stop_epoch, params, logger=val_logger, trn_logger=trn_logger)
