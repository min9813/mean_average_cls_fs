# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, debug, n_way, n_support, output_dim, del_last_relu, normalize_method="no", norm_factor=1., normalize_vector="no", normalize_query=True, subtract_mean=False):
        super(ProtoNet, self).__init__(model_func,  n_way,
                                       n_support, normalize_method=normalize_method,
                                       normalize_vector=normalize_vector,
                                       subtract_mean=subtract_mean,
                                       output_dim=output_dim,
                                       del_last_relu=del_last_relu,
                                       normalize_query=normalize_query)
        self.loss_fn = nn.CrossEntropyLoss()

        self.test_require_grad_parts = []
        self.debug = debug
        self.norm_factor = norm_factor
        self.normalize_vector = normalize_vector
        self.normalize_query = normalize_query

    def set_forward(self, x, is_feature=False, is_cuda=True, normalize_feature=False):
        z_support, z_query = self.parse_feature(
            x, is_feature, normalize_feature=normalize_feature)

        z_support = z_support.contiguous()
        # the shape of z is [n_data, n_dim]
        z_support = z_support.view(self.n_way, self.n_support, -1)

        z_support = z_support * self.norm_factor
        z_query = z_query * self.norm_factor

        z_proto = z_support.mean(1)

        if self.normalize_method == "l2":
            if self.normalize_vector in ("mean", "before_and_mean"):
                z_proto = z_proto / \
                    torch.sqrt(torch.sum(z_proto*z_proto, dim=1, keepdim=True))
                z_proto = z_proto * self.norm_factor

        # query_norm  = torch.sqrt(torch.sum(z_query*z_query, dim=2, keepdim=True))
        # support_norm  = torch.sqrt(torch.sum(z_support*z_support, dim=2, keepdim=True))
        # mean_norm  = torch.sqrt(torch.sum(z_proto*z_proto, dim=1, keepdim=True))
        # print("query norm")
        # print(query_norm)

        # print("support norm")
        # print(support_norm)

        # print("mean_norm")
        # print(mean_norm)
        # sfda
            # z_query = z_query * self.norm_factor
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # print(torch.sum(z_query*z_query, dim=1))
        # sfda
        if "maha" in self.normalize_method:
            if "diag_cov" in self.normalize_method:
                D = z_support.shape[2]
                z_var = torch.var(z_support, unbiased=False, dim=1)

                z_var_all = torch.var(
                    z_support.reshape(-1, D), dim=0, keepdim=True)
                z_var_all = z_var_all.expand(len(z_var), z_var_all.shape[1])
                # # # cov
                mask = torch.abs(z_var) < 1e-8
                z_var[mask] = z_var_all[mask]
                dists = calc_mahalanobis_torch(
                    feature1=z_query,
                    feature2=z_proto,
                    sigma=z_var
                )
            else:
                class_cov_list = []
                for i in range(self.n_way):
                    class_vector = z_support[i]
                    class_cov = cov(class_vector)
                    class_cov_list.append(class_cov)

                class_cov_tensor = torch.stack(class_cov_list, dim=0)

                dists = calc_mahalanobis_not_diag(
                    feature1=z_query,
                    feature2=z_proto,
                    sigma=class_cov_tensor
                )

        else:
            dists = euclidean_dist(z_query, z_proto)

        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def calc_mahalanobis_torch(feature1, feature2, sigma):
    # XX = feature1 * feature1
    # XXs = XX[:, None, :] / sigma[None, :, :]
    # XXs = torch.sum(XXs, dim=2)
    # XY = torch.mm(feature1, (feature2 / sigma).T)
    # YY = (feature2 * feature2 / sigma).sum(dim=1)
    # dist = XXs - 2*XY + YY
    n = feature1.size(0)
    m = feature2.size(0)
    d = feature1.size(1)
    assert d == feature2.size(1)
    assert d == sigma.size(1)
    assert m == sigma.size(0)

    feature1 = feature1.unsqueeze(1).expand(n, m, d)
    feature2 = feature2.unsqueeze(0).expand(n, m, d)

    sigma = sigma.unsqueeze(0).expand(n, m, d)

    diff = (feature1 - feature2)
    dist = (diff * diff) / sigma
    dist = torch.sum(dist, dim=2)
    # print(dist.shape)
    # print(feature1.shape)
    # print(feature2.shape)

    return dist


def calc_mahalanobis_not_diag(feature1, feature2, sigma):
    """
    feature1: N, D
    feature2: n_way, D
    sigma: n_way, D, D
    """

    n_way = feature2.shape[0]
    assert n_way == sigma.shape[0], (n_way, sigma.shape)

    dist_to_class_list = []
    for i in range(n_way):
        class_vector = feature2[[i]]
        diff = feature1 - class_vector

        class_sigma = sigma[i]
        class_sigma = torch.inverse(class_sigma)

        dist_to_class = torch.sum(torch.matmul(
            diff, class_sigma) * diff, dim=1)
        dist_to_class_list.append(dist_to_class)

    dist_to_class_list = torch.stack(dist_to_class_list, dim=1)

    return dist_to_class_list


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
