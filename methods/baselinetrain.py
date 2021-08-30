import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type='softmax', debug=False, del_last_relu=False, output_dim=512):
        super(BaselineTrain, self).__init__()
        self.feature = model_func(del_last_relu=del_last_relu, output_dim=output_dim)
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(
                self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.DBval = False  # only set True for CUB dataset, see issue #31
        self.debug = debug

    def forward(self, x):
        x = x.cuda()
        out = self.feature.forward(x)
        # print(out.shape)
        # print(out.shape)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer, trn_logger, params):
        print_freq = 10
        if self.debug:
            print_freq = 1
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # print(x.shape)
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()



            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                trn_logger.info('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        i, len(train_loader), avg_loss/float(i+1)))

            if self.debug:
                if i >= 5:
                    break

    def test_loop(self, val_loader, n_way=5, n_query=15, n_support=5, is_aug=False, params=None):
        if self.DBval:
            return self.analysis_loop(val_loader)
        else:
            if is_aug:
                raise NotImplementedError
            else:
                acc_list = self.finetune_loop(val_loader=val_loader, n_way=n_way, n_support=n_support, n_query=n_query) # no validation, just save model during iteration
                return acc_list

    def finetune_loop(self, val_loader, n_way=5, n_support=5, n_query=15, n_episode=600):
        class_file = {}
        for i, (x, y) in enumerate(val_loader):
            # x_var = Variable(x)
            x_var = x.cuda()
            # print(x.shape)
            feats = self.feature.forward(x_var).data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(feats, labels):
                if l not in class_file.keys():
                    class_file[l] = []
                class_file[l].append(f)

            if self.debug:
                if i >= 10:
                    break

        for cl in class_file:
            class_file[cl] = np.array(class_file[cl])

        class_list = list(class_file.keys())
        acc_list = []
        for i in tqdm(range(n_episode)):
            sample_classes = np.random.choice(
                class_list, size=n_way, replace=False)
            # episode_class_feats = {}
            f_support = []
            f_query = []
            for _class in sample_classes:
                each_class_feats = class_file[_class]
                if len(each_class_feats) < n_support + n_query:
                    # pick_indices = pick_indices[:n_support+n_query]
                    pick_indices = np.random.choice(range(len(each_class_feats)), size=n_support+n_query, replace=True)

                else:
                    pick_indices = np.random.permutation(len(each_class_feats))
                    pick_indices = pick_indices[:n_support+n_query]

                pick_class_feats = each_class_feats[pick_indices]
                # episode_class_feats[_class] = pick_class_feats
                f_support.append(pick_class_feats[:n_support])
                f_query.append(pick_class_feats[n_support:])

            f_support = np.array(f_support)
            f_query = np.array(f_query)

            # print(f_query.shape)
            # print(f_support.shape)

            acc = self.test_one_episode(z_support=f_support, z_query=f_query, n_way=n_way, n_query=n_query, n_support=n_support)
            acc_list.append(acc)

            if self.debug:
                if i >= 10:
                    break

        return acc_list

    def test_one_episode(self, z_support, z_query, n_way, n_support, n_query):
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        z_support = z_support.reshape(self.n_way * self.n_support, -1)
        z_query = z_query.reshape(self.n_way * self.n_query, -1)

        z_support = torch.from_numpy(z_support)
        z_query = torch.from_numpy(z_query)

        y_support = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support))
        # y_support = Variable(y_support.cuda())
        y_support = y_support

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feature.final_feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feature.final_feat_dim, self.n_way)

        device = "cpu"
        linear_clf = linear_clf.to(device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(
        ), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.cuda()
        y_support = y_support.to(device)
        z_support = z_support.to(device)

        z_query = z_query.to(device)
        
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            # rand_id = np.random.permutation(support_size)
            rand_id = torch.randperm(support_size, device=z_support.device)
            # print(rand_id)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                # selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                # selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ])
                selected_id = rand_id[i: min(i+batch_size, support_size)]
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.n_way), self.n_query)
        acc = np.mean(pred == y)*100 
        return acc

    def parse_feature(self, x, is_feature, is_cuda=True):
        if is_cuda:
            x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(
                self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def analysis_loop(self, val_loader, record=None):
        class_file = {}
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var).data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(feats, labels):
                if l not in class_file.keys():
                    class_file[l] = []
                class_file[l].append(f)

        for cl in class_file:
            class_file[cl] = np.array(class_file[cl])

        DB = DBindex(class_file)
        print('DB index = %4.2f' % (DB))
        return 1/DB  # DB index: the lower the better


def DBindex(cl_data_file):
    # For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    # DB index present the intra-class variation of the data
    # As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    # Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(
            np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0),
                   (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j])/mdists[i, j]
                           for j in range(cl_num) if j != i]))
    return np.mean(DBs)
