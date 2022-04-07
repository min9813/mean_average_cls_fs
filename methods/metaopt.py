from cProfile import label
from methods import meta_test_preprocess
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
from metaopt_models import classification_heads


class MetaOpt(nn.Module):

    def __init__(self, model_func, n_way, n_support, subtract_mean=False, add_final_layer=False, normalize_method="no", normalize_vector="no", normalize_query=True, del_last_relu=False, output_dim=512, debug=False):
        super(MetaOpt, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.feature = model_func(
            del_last_relu=del_last_relu, output_dim=output_dim, add_final_layer=add_final_layer)

        self.feature = self.feature.cuda()

        self.cls_head = classification_heads.ClassificationHead(
            base_learner="SVM-CS"
        )
        self.cls_head = self.cls_head.cuda()
        self.debug = debug
        self.support_aug_num = 1

        self.cls_loss = nn.CrossEntropyLoss()


    def forward(self, x):
        x = x.cuda()

        features = self.feature(x)
        return features

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

        return z_support, z_query

    def set_forward(self, x, is_feature, is_cuda=True):
        z_support, z_query = self.parse_feature(x, is_feature=is_feature)

        # label_for_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support).reshape(self.n_way, self.n_support))
        label_for_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).reshape(1, -1)
        label_for_support = label_for_support.cuda()

        n_way, n_query = z_query.shape[:2]

        z_support = z_support.reshape(1, self.n_way*self.n_support*self.support_aug_num, -1)
        z_query = z_query.reshape(1, self.n_way*n_query*self.support_aug_num, -1)
        # label_for_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_support).reshape(self.n_way, self.n_support))
        # label_for_query = label_for_query.cuda()

        scores = self.cls_head(query=z_query, support=z_support, support_labels=label_for_support, n_way=self.n_way, n_shot=self.n_support)

        return scores

        
    def correct(self, x):
        scores = self.set_forward(x, is_feature=len(x.shape) < 4, is_cuda=True)
        if scores.shape[1] > self.n_way:
            y_query = np.repeat(range(self.n_way), self.n_query)

        else:
            y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        # print(topk_ind.shape)
        # sfa
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=len(x.shape) < 4, is_cuda=True)
        # if scores.shape[1] > self.n_way:
        n_query = scores.shape[1] // self.n_way
        y_query = torch.arange(self.n_way, dtype=torch.long, device=scores.device).reshape(self.n_way,1).repeat(1, n_query)
        y_query = y_query.reshape(-1)
        # print(scores.shape, y_query.shape)
        scores = scores.reshape(-1, self.n_way)

        loss = self.cls_loss(scores, y_query)
        return loss

    def train_loop(self, epoch, train_loader, optimizer, trn_logger=None, params=None):
        print_freq = 10

        if trn_logger is None:
            print_ = print

        else:
            print_ = trn_logger.info

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print_('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(
                    epoch,
                    i,
                    len(train_loader),
                    avg_loss/float(i+1)))

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