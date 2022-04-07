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
import copy
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from methods import lda


def adjust_to_class_mean_norm(cl_data_file):
    for label, feature_list in cl_data_file.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)

        # print(feature_list.shape)
        # sdfa

        mean_feature = np.mean(feature_list, axis=0, keepdims=True)
        mean_norm = np.sqrt(
            np.sum(mean_feature * mean_feature, axis=1, keepdims=True))

        feature_norms = np.sqrt(
            np.sum(feature_list * feature_list, axis=1, keepdims=True))
        mean_of_norm = np.mean(feature_norms)
        # std_of_norm = np.std(feature_norms)
        # print(label, mean_norm, mean_of_norm, std_of_norm)
        feature_list = feature_list / feature_norms * mean_of_norm

        feature_norms = np.sqrt(
            np.sum(feature_list * feature_list, axis=1, keepdims=True))
        # print(feature_norms)

        cl_data_file[label] = feature_list
    # sdfa

    return cl_data_file


def minimize_variance_of_norm_by_trace_torch(z_support, z_query):
    """
    z_support: (n_way, n_support, dim)
    """
    n_way, n_support, dim = z_support.shape
    n_way_q, n_query, dim = z_query.shape

    each_class_mean_vector = torch.mean(z_support, dim=1)

    # z_support = torch.cat((z_support, z_query), dim=1)
    # z_support = z_support.reshape(n_way*(n_support+n_query), dim)
    z_support = z_support.reshape(n_way*(n_support), dim)
    vvt = torch.bmm(
        z_support[:, :, None],
        z_support[:, None, :]
    )
    # vvt = vvt.reshape(n_way, n_support+n_query, dim, dim)
    vvt = vvt.reshape(n_way, n_support, dim, dim)

    vvt_mean = torch.bmm(
        each_class_mean_vector[:, :, None],
        each_class_mean_vector[:, None, :],
    )
    vvt_mean = vvt_mean.reshape(n_way, 1, dim, dim)

    # print(vvt.shape)
    # print(vvt_mean.shape)

    variance_of_norm = vvt - vvt_mean  # (n_way, n_support, dim, dim)
    variance_of_norm = torch.mean(variance_of_norm * variance_of_norm, dim=1)

    variance_of_norm = torch.mean(variance_of_norm, dim=0)
    variance_of_norm_reg = variance_of_norm + \
        torch.eye(dim, device=variance_of_norm.device,
                  dtype=variance_of_norm.dtype) * 1e-2

    w, v = torch.symeig(variance_of_norm_reg, eigenvectors=True)

    operate_target = w > 1e-2
    operate_w = w[operate_target]

    adjust_coef = 1 / operate_w
    adjust_coef = adjust_coef / torch.sum(adjust_coef)
    # new_a_
    adjust_coef_new = torch.ones_like(w)
    adjust_coef_new[operate_target] = adjust_coef

    new_matrix_A = torch.matmul(v, torch.matmul(
        adjust_coef_new.diag_embed(), v.transpose(-2, -1)))

    # adjust_coef_new_square = adjust_coef_new * adjust_coef_new
    # new_matrix_A_square =
    # new_matrix_A_square = torch.matmul(v, torch.matmul(adjust_coef_new_square.diag_embed(), v.transpose(-2, -1)))

    # transformed_variance = torch.matmul(new_matrix_A_square, variance_of_norm)
    # trace_transformed = torch.trace(transformed_variance)
    # trace_normal = torch.trace(variance_of_norm)
    # print(trace_transformed, trace_normal)
    # sfda
    # z_support = torch.matmul(new_matrix_A, )
    # dsfa

    z_support_transformed = torch.matmul(
        z_support, new_matrix_A.T
    )
    # z_support_transformed = z_support_transformed.reshape(
    # n_way, n_support+n_query, dim)
    z_support_transformed = z_support_transformed.reshape(
        n_way, n_support, dim)
    z_support_transformed = z_support_transformed[:, :n_support]

    z_query = z_query.reshape(n_way_q*n_query, dim)
    z_query_transformed = torch.matmul(
        z_query, new_matrix_A.T
    )
    z_query_transformed = z_query_transformed.reshape(n_way_q, n_query, dim)

    z_proto = torch.mean(z_support_transformed, dim=1)

    # z_support = z_support.reshape(n_way, n_support+n_query, dim)
    z_support = z_support.reshape(n_way, n_support, dim)
    z_support = z_support[:, :n_support]

    z_support_norm = torch.sqrt(torch.sum(z_support*z_support, dim=-1))
    z_support_transformed_norm = torch.sqrt(
        torch.sum(z_support_transformed*z_support_transformed, dim=-1))

    # print(z_support_norm)
    # print(torch.var(z_support_norm, dim=1))
    # print(z_support_transformed_norm)
    # print(torch.var(z_support_transformed_norm, dim=1))

    support_transformed_norm_mean = torch.mean(z_support_transformed_norm)

    z_query = z_query.reshape(n_way_q, n_query, dim)
    query_norm = torch.sqrt(torch.sum(z_query*z_query, dim=2, keepdim=True))

    z_query_transformed = z_query_transformed.reshape(n_way_q, n_query, dim)
    query_transformed_norm = torch.sqrt(
        torch.sum(z_query_transformed*z_query_transformed, dim=2, keepdim=True))
    query_transformed_norm_mean = torch.mean(query_transformed_norm)

    # z_query_transformed = z_query_transformed / query_transformed_norm * support_transformed_norm_mean
    # print(query_norm)
    # print(query_transformed_norm)
    # sfa

    return z_support_transformed, z_query_transformed, z_proto


def minimize_variance_beforehand(base_features, novel_features):
    base_feature_all = []
    for label, feature_list in base_features.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)[:300]

        base_feature_all.append(feature_list)

    base_feature_all = np.array(base_feature_all)
    base_feature_all = torch.from_numpy(base_feature_all)

    novel_feature_all = []
    for label, feature_list in novel_features.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)

        novel_feature_all.append(feature_list)

    novel_feature_all = np.array(novel_feature_all)
    novel_feature_all = torch.from_numpy(novel_feature_all)

    z_support_transformed, z_query_transformed, z_proto = minimize_variance_of_norm_by_trace_torch(
        z_support=base_feature_all,
        z_query=novel_feature_all
    )

    novel_features_new = {}
    for i in range(z_query_transformed.shape[0]):
        novel_features_new[i] = z_query_transformed[i].tolist()

    return novel_features_new


def est_test(support_features, query_features=None, rho=0.001, pick_dim=60):
    """
    z_support: (n_way, n_support, dim)
    """

    n_way, n_support, dim = support_features.shape
    all_feature_means = torch.mean(
        support_features.reshape(-1, dim), dim=0, keepdim=True)

    each_class_means = torch.mean(support_features, dim=1, keepdim=True)
    diff_to_all_means = each_class_means[:, 0] - all_feature_means
    inter_class_cov = torch.bmm(
        diff_to_all_means[:, :, None], diff_to_all_means[:, None, :])

    inter_class_cov = torch.mean(inter_class_cov, dim=0)

    each_class_cov_list = []

    for i in range(n_way):
        each_class_feats = support_features[i]
        diff_to_mean = each_class_feats - each_class_means[i]
        class_intra_cov = torch.bmm(
            diff_to_mean[:, :, None], diff_to_mean[:, None, :])
        class_intra_cov = torch.mean(class_intra_cov, dim=0)

        each_class_cov_list.append(class_intra_cov)

    each_class_cov_list = torch.stack(each_class_cov_list, dim=0)
    each_class_cov_list = torch.mean(each_class_cov_list, dim=0)

    est_mat = inter_class_cov - rho * each_class_cov_list
    w, v = torch.symeig(est_mat, eigenvectors=True)

    # w_mask = w > 1e-2
    # print(w.shape)
    pick_w_indices = torch.argsort(w)
    # print(pick_w_indices)
    pick_w_indices = pick_w_indices[-pick_dim:]
    # print(pick_w_indices)
    # sfa
    masked_w = w[pick_w_indices]
    pick_v = v[:, pick_w_indices]

    if query_features is not None:
        n_way_q, n_query, dim = query_features.shape
        query_features = query_features.reshape(n_way_q * n_query, dim)
        # print(query_features.shape)
        query_features = torch.mm(query_features, pick_v)
        # print(query_features.shape)
        # print(pick_v.shape)

        query_features = query_features.reshape(
            n_way_q, n_query, pick_v.shape[1])

        support_features = torch.mm(support_features.reshape(-1, dim), pick_v)
        support_features = support_features.reshape(
            n_way, n_support, pick_v.shape[1])

        return w, pick_v, support_features, query_features

    else:
        return w, masked_w, pick_v


def est_beforehand(base_features, novel_features, get_nd_array=False):
    # print
    base_feature_all = []
    min_len = None
    for label, feature_list in base_features.items():
        if min_len is None:
            min_len = len(feature_list)

        else:
            min_len = min(min_len, len(feature_list))

    for label, feature_list in base_features.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)
        # print(feature_list.shape, feature_list.dtype)
        # feature_list = feature_list.astype(np.float32)
        feature_list = feature_list[:min_len]

        # print(feature_list.shape, feature_list.dtype)

        base_feature_all.append(feature_list)

    base_feature_all = np.array(base_feature_all).astype(np.float32)
    base_feature_all = torch.from_numpy(base_feature_all)

    min_len = None
    for label, feature_list in novel_features.items():
        if min_len is None:
            min_len = len(feature_list)

        else:
            min_len = min(min_len, len(feature_list))

    novel_feature_all = []
    for label, feature_list in novel_features.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)

        feature_list = feature_list[:min_len]
        novel_feature_all.append(feature_list)

    novel_feature_all = np.array(novel_feature_all)
    novel_feature_all = torch.from_numpy(novel_feature_all)

    _, _, z_support_transformed, z_query_transformed = est_test(
        support_features=base_feature_all,
        query_features=novel_feature_all,
        rho=0.001
    )

    novel_features_new = {}
    for i in range(z_query_transformed.shape[0]):
        if get_nd_array:
            novel_features_new[i] = z_query_transformed[i].numpy()

        else:
            novel_features_new[i] = z_query_transformed[i].tolist()

    base_feature_new = {}
    # print(z_support_transformed.shape)
    # sfda
    for i in range(z_support_transformed.shape[0]):
        if get_nd_array:
            base_feature_new[i] = z_support_transformed[i].numpy()

        else:
            base_feature_new[i] = z_support_transformed[i].tolist()

    return novel_features_new, base_feature_new


def lda_test(support_features, query_features):
    # support_labels =
    support_features = support_features.cpu()
    query_features = query_features.cpu()

    n_way_q, n_query, dim = query_features.shape
    query_features = query_features.reshape(-1, dim)

    _, logit, pick_v = lda.lda_for_episode(
        support_vector=support_features,
        query_vector=query_features,
        lamb=0.0001,
    )

    n_way, n_support, dim = support_features.shape
    support_features = support_features.reshape(-1, dim)

    query_features = torch.mm(query_features, pick_v)
    support_features = torch.mm(support_features, pick_v)

    support_features = support_features.reshape(
        n_way, n_support, pick_v.shape[1])
    query_features = query_features.reshape(n_way_q, n_query, pick_v.shape[1])

    return logit, support_features, query_features


def l2_normalize_all(cl_data_file):
    for label, feature_list in cl_data_file.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)[:300]

        feature_norm = np.sqrt(
            np.sum(feature_list*feature_list, axis=1, keepdims=True))
        feature_list = feature_list / feature_norm

        cl_data_file[label] = feature_list

    return cl_data_file

def zscore_normalize_all(cl_data_file):
    for label, feature_list in cl_data_file.items():
        if isinstance(feature_list, list):
            feature_list = np.array(feature_list)[:300]

        mean_values = np.mean(feature_list, axis=1, keepdims=True)
        std_values = np.std(feature_list, axis=1, keepdims=True)

        feature_list = (feature_list - mean_values) / std_values
        # feature_norm = np.sqrt(
            # np.sum(feature_list*feature_list, axis=1, keepdims=True))
        # feature_list = feature_list / feature_norm

        cl_data_file[label] = feature_list

    return cl_data_file


def calc_center_point(feature_info, method="mean"):
    for label, class_features in feature_info.items():
        if not isinstance(class_features, np.ndarray):
            class_features = np.array(class_features)

        feature_info[label] = class_features

    if method == "mean":
        all_mean_feats = 0
        for label, class_features in feature_info.items():
            class_means = np.mean(class_features, axis=0, keepdims=True)
            all_mean_feats += class_means

        all_mean_feats = all_mean_feats / len(feature_info)

    elif "cov_normalize" in method:
        class2stats = {}
        for label, class_features in feature_info.items():
            class_means = np.mean(class_features, axis=0, keepdims=True)
            class_cov = np.cov(class_features.T)

            class2stats[label] = {
                "mean_vec": class_means,
                "within_cov": class_cov
            }

        inversed_cov_sum = 0
        transformed_mean_vec = 0
        for label, stats in tqdm(class2stats.items()):
            #     print(stats.keys())
            #     print(within_class_covariance.jpeg)
            #     cov_inverse = np.linalg.inv(stats["within_cov"])
            mean_vec = stats["mean_vec"]
            N = stats["within_cov"].shape[0]
            indices = np.arange(N)

            if "diag" in method:
                diag_values = stats["within_cov"][indices, indices]
                inversed_diag_values = 1. / diag_values[None, :]
                inversed_mean_vec = mean_vec * inversed_diag_values
                inversed_cov_sum += inversed_diag_values

            else:
                cov_inverse = np.linalg.inv(stats["within_cov"])
                inversed_mean_vec = mean_vec @ cov_inverse
                inversed_cov_sum += cov_inverse

            mean_vec = stats["mean_vec"]
        #     inversed_mean_vec = mean_vec @ cov_inverse
#             inversed_mean_vec = mean_vec * diag_values[None, :]
            transformed_mean_vec += inversed_mean_vec

        if "diag" in method:
            inversed_cov_sum_inverse = 1 / inversed_cov_sum
            all_mean_feats = transformed_mean_vec * inversed_cov_sum_inverse

        else:
            inversed_cov_sum_inverse = np.linalg.inv(inversed_cov_sum)
            all_mean_feats = transformed_mean_vec @ inversed_cov_sum_inverse

    elif "equal_angle" in method:
        class2stats = {}
        for label, class_features in feature_info.items():
            class_means = np.mean(class_features, axis=0, keepdims=True)
            class_cov = np.cov(class_features.T)

            class2stats[label] = {
                "mean_vec": class_means,
                "within_cov": class_cov
            }
        all_mean_vec = []
#         all_vectors = []
        label2all_vectors = {}
        for label, stats in tqdm(class2stats.items()):
            mean_vec = stats["mean_vec"]
            mean_vec = torch.from_numpy(mean_vec.astype(np.float32))
            all_mean_vec.append(mean_vec)

            each_class_features = feature_info[label]
#             all_vectors.append()
            label2all_vectors[label] = torch.from_numpy(
                each_class_features.astype(np.float32))

        all_mean_vec = torch.cat(all_mean_vec, dim=0)
#         all_mean_vec.requires_grad = True
#         all_mean_vec.retain_grad()
        mean_vector = torch.mean(all_mean_vec, axis=0, keepdim=True)

#         center_vec = torch.nn.Parameter(mean_vector)
        if "mean_start" in method:
            center_vec = torch.nn.Parameter(mean_vector.clone())
        elif "zero_start" in method:
            center_vec = torch.nn.Parameter(torch.zeros_like(mean_vector))
#         center_vec = mean_vector
#         center_vec.requires_grad = True
#         center_vec.retain_grad()

        lr = 0.01
        prev_loss = 0
        for i in range(100):
            mean_inter_cov = 0

            all_class_mean_vectors = []
            for label, each_class_features in label2all_vectors.items():
                each_class_features = each_class_features - center_vec
                each_class_features = each_class_features / \
                    torch.sqrt(torch.sum(each_class_features *
                                         each_class_features, dim=1, keepdim=True)) * 8

                class_mean_vector = torch.mean(
                    each_class_features, dim=0, keepdim=True)

                diff_to_class_center = each_class_features - class_mean_vector
                trace_intra_cov = torch.sum(
                    diff_to_class_center * diff_to_class_center, dim=1).mean()

                all_class_mean_vectors.append(class_mean_vector)
#                 mean_inter_cov.append(trace_intra_cov)
                mean_inter_cov = trace_intra_cov / \
                    len(label2all_vectors) + mean_inter_cov
            # print(each_class_features.grad)

            all_class_mean_vectors = torch.cat(all_class_mean_vectors)
            mean_transformed_vector = torch.mean(
                all_class_mean_vectors, dim=0, keepdim=True)
            diff_to_mean = all_class_mean_vectors - mean_transformed_vector

            trace_inter_cov = torch.sum(
                diff_to_mean * diff_to_mean, dim=1).mean()

#             mean_inter_cov = torch.cat(mean_inter_cov, dim=0)
#             mean_inter_cov = torch.mean(mean_inter_cov)

            # ratio = - trace_inter_cov / mean_inter_cov
            # loss = ratio
#             loss = mean_inter_cov

#             distance_matrix = calc_l2_dist_torch(feature1=diff_vector, feature2=diff_vector, is_neg=False)
#             loss =  - torch.mean(distance_matrix)
            loss = - trace_inter_cov + mean_inter_cov

            if (torch.abs(loss - prev_loss).item() < 1e-4):
                break
#             loss = - trace_inter_cov

#             loss.backward()
            gradients = torch.autograd.grad(
                loss, center_vec)[0]

            center_vec = center_vec - lr * gradients
            # print(gradients)
            # print(loss, trace_inter_cov, mean_inter_cov)

        all_mean_feats = center_vec.detach().numpy()
        mean_vector = mean_vector.numpy()

        moved_norm = all_mean_feats - mean_vector
        moved_norm = np.sum(moved_norm * moved_norm)
        print("moving norm : {:.4f}".format(moved_norm))

    else:
        raise NotImplementedError

    return all_mean_feats
