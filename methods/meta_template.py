import os
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
import cv2
from abc import abstractmethod
from tqdm import tqdm


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way=True, subtract_mean=False, normalize_method="no", normalize_vector="no", normalize_query=True, del_last_relu=False, output_dim=512):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func(
            del_last_relu=del_last_relu, output_dim=output_dim)
        self.feat_dim = self.feature.final_feat_dim
        # some methods allow different_way classification during training and test
        self.change_way = change_way
        self.support_aug_num = 1

        assert normalize_method in ("no", "l2", "maha", "maha-diag_cov")
        assert normalize_vector in ("before", "mean", "before_and_mean", "no")

        self.normalize_method = normalize_method
        self.normalize_vector = normalize_vector
        self.normalize_query = normalize_query
        self.subtract_mean = subtract_mean

    @abstractmethod
    def set_forward(self, x, is_feature, normalize_feature=False):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature, is_cuda=True, is_detach=False, normalize_feature=False):
        if is_cuda:
            x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query)*self.support_aug_num, *x.size()[2:])

            if is_detach:
                with torch.no_grad():
                    z_all = self.feature.forward(x)

            else:
                z_all = self.feature.forward(x)

            z_all = z_all.view(
                self.n_way, (self.n_support + self.n_query)*self.support_aug_num, -1)

            # z_all = z_all / torch.sqrt(torch.sum(z_all*z_all, dim=2, keepdim=True))

        # if self.subtract_mean:
        #     dim = z_all.shape[-1]
        #     z_all_mean = torch.mean(z_all.view(-1, dim).contiguous(), dim=0)

        #     z_all = z_all - z_all_mean[None, None, :]

        z_support = z_all[:, :self.n_support*self.support_aug_num]
        z_query = z_all[:, self.n_support*self.support_aug_num:]

        if self.normalize_method == "l2":
            if self.normalize_query:
                z_query = z_query / \
                    torch.sqrt(
                        torch.sum(z_query * z_query, dim=2, keepdim=True))

            if self.normalize_vector in ("before_and_mean", "before"):
                z_support = z_support / \
                    torch.sqrt(
                        torch.sum(z_support * z_support, dim=2, keepdim=True))

            # z_support = z_support * self.norm_factor

        # print("z_query before shape:", z_query.shape)
        if self.support_aug_num > 1:
            z_query = z_query[:, ::self.support_aug_num]

        # print("z_query shape:", z_query.shape)
        # print("z_support shape:", z_support.shape)

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x, is_feature=len(x.shape) < 4, is_cuda=True)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer, trn_logger=None, params=None):
        print_freq = 10

        if trn_logger is None:
            print_ = print

        else:
            print_ = trn_logger.info

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print_('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                         i, len(train_loader), avg_loss/float(i+1)))

            if self.debug and i >= 5:
                break

    def test_loop(self, test_loader, params, record=None, return_std=True, is_aug=False):
        correct = 0
        count = 0
        acc_all = []

        if params.support_aug:
            self.support_aug_num = params.support_aug_num

        else:
            self.support_aug_num = 1

        if not hasattr(self, "test_require_grad_parts"):
            self.test_require_grad_parts = []

        iter_num = len(test_loader)
        self.mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]

        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="test"):
            if not isinstance(data, dict):
                x, _ = data

            else:
                x_list = []
                for i in range(params.support_aug_num):
                    x_list.append(data["image_{}".format(i)])

                x = torch.stack(x_list, dim=2)
                B, NB, NA, C, W, H = x.shape
                x = x.reshape(B, NB*NA, C, W, H)

                # for i in range(x.shape[0]):
                #     class_tensor = x[i]
                #     class_tensor = ((class_tensor * self.std) + self.mean) * 255

                #     class_tensor = class_tensor.numpy().astype(np.uint8)
                #     class_tensor = class_tensor.transpose(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
                #     class_tensor = np.concatenate(class_tensor, axis=1)
                #     print(class_tensor.shape)
                #     os.makedirs("./check", exist_ok=True)
                #     cv2.imwrite("./check/aug_data_{}.jpeg".format(i), class_tensor)
                # sdfa

            self.n_query = (x.size(1) - self.n_support *
                            self.support_aug_num) // self.support_aug_num
            if self.change_way:
                self.n_way = x.size(0)

            if len(self.test_require_grad_parts) == 0:
                with torch.no_grad():
                    correct_this, count_this = self.correct(x)

            else:
                correct_this, count_this = self.correct(x)

            acc_all.append(correct_this / count_this*100)

            if self.debug and i >= 5:
                break

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num,  acc_mean, 1.96 * acc_std/np.sqrt(iter_num)))

        return acc_all, acc_mean, acc_std

    # further adaptation, default is fixing feature and train a new softmax clasifier
    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(
        ), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(
                    rand_id[i: min(i+batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores
