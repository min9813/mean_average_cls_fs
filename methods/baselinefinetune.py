from numpy.lib.function_base import select
from torch._C import device
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type="softmax", normalize_method="no", del_last_relu=False, output_dim=512, subtract_mean=False):
        super(BaselineFinetune, self).__init__(
            model_func,  n_way, n_support, 
            normalize_method=normalize_method, 
            del_last_relu=del_last_relu, 
            output_dim=output_dim,
            subtract_mean=subtract_mean
            )
        self.loss_type = loss_type

    def set_forward(self, x, is_feature=True, is_cuda=False, normalize_feature=False):
        # Baseline always do adaptation
        return self.set_forward_adaptation(x, is_feature, is_cuda=is_cuda, normalize_feature=normalize_feature)

    def set_forward_adaptation(self, x, is_feature=True, is_cuda=False, normalize_feature=False):
        # assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query = self.parse_feature(x, is_feature, is_cuda=is_cuda, is_detach=True, normalize_feature=normalize_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support * self.support_aug_num, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        is_cuda = False
        if is_cuda:
            device = "cuda"

        else:
            device = "cpu"

        device = torch.device(device)

        # y_support = torch.from_numpy(
            # np.repeat(range(self.n_way), self.n_support*self.support_aug_num))
        # print(y_support)
        y_support = torch.arange(self.n_way, device=device)[:, None].repeat(1, self.n_support*self.support_aug_num).reshape(-1)
        # print(y_support)
        # sdfa
        # y_support = Variable(y_support.cuda())
        # y_support = y_support

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)

        linear_clf = linear_clf.to(device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(
        ), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.cuda()
        y_support = y_support.to(device)
        z_support = z_support.to(device)

        z_query = z_query.to(device)

        batch_size = 4
        support_size = self.n_way * self.n_support * self.support_aug_num
        for epoch in range(100):
            # rand_id = np.random.permutation(support_size)
            rand_id = torch.randperm(support_size, device=device)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                # selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                # selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ])
                selected_id = rand_id[i: min(i+batch_size, support_size)]
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self, x):
        raise ValueError(
            'Baseline predict on pretrained feature and do not support finetune backbone')
