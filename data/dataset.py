# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import io_utils


def identity(x): return x


def transform_pickle_format(pickle_data):
    """
    pickle_data: {
        `data`: (num_image, 32, 32, 3),
        `labels`: (num_image,)
    }

    return data: {
        image_names: (num_image, 32, 32, 3),
        image_labels: (num_image,)
    }
    """

    new_data = {}
    new_data["image_names"] = pickle_data["data"]
    new_data["image_labels"] = pickle_data["labels"]

    unique_labels = np.unique(new_data["image_labels"])
    data_label2train_label = {}
    for i, label in enumerate(sorted(unique_labels)):
        data_label2train_label[label] = i

    return new_data, data_label2train_label


def load_all_data(json_data):
    """
    data: {
        image_names: [image_path]* num_image,
        image_labels: [image_label] * num_image
    }
    """

    images = []
    for image_path in json_data["image_names"]:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        images.append(img)

    images = np.stack(images, axis=0)

    new_data = {
        "image_names": images,
        "image_labels": json_data["image_labels"]
    }

    return new_data


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        if data_file.endswith(".pickle"):
            data = io_utils.load_cifar_pickle(
                pickle_path=data_file
            )
            self.meta, data_label2train_label = transform_pickle_format(
                pickle_data=data)
            self.is_loaded = True
            self.data_label2train_label = data_label2train_label

            self.target_transform = lambda x: self.data_label2train_label[x]

        else:
            with open(data_file, 'r') as f:
                self.meta = json.load(f)

            if "cifar" in data_file.lower():
                self.meta = load_all_data(self.meta)
                self.is_loaded = True

            else:
                self.is_loaded = False

            self.target_transform = target_transform

        self.transform = transform

    def __getitem__(self, i):
        if self.is_loaded:
            img = self.meta["image_names"][i]
            img = Image.fromarray(img)

        else:
            image_path = os.path.join(self.meta['image_names'][i])
            img = Image.open(image_path).convert('RGB')

        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class AugDataset(SimpleDataset):

    def __init__(self, data_file, transform, target_transform=identity, aug_num=5):
        super(AugDataset, self).__init__(data_file=data_file,
                                         transform=transform, target_transform=target_transform)
        self.aug_num = aug_num

    def __getitem__(self, i):
        # image_path = os.path.join(self.meta['image_names'][i])
        # img_original = Image.open(image_path).convert('RGB')
        if self.is_loaded:
            img = self.meta["image_names"][i]
            img_original = Image.fromarray(img)

        else:
            image_path = os.path.join(self.meta['image_names'][i])
            img_original = Image.open(image_path).convert('RGB')

        data = {}
        for i in range(self.aug_num):
            img = self.transform(img_original)
            data["image_{}".format(i)] = img

        target = self.target_transform(self.meta['image_labels'][i])
        data["label"] = target

        return data


class SetDataset:
    def __init__(self, data_file, batch_size, transform, normal_transform=None, support_aug=False, support_aug_num=5):
        if data_file.endswith(".pickle"):
            data = io_utils.load_cifar_pickle(
                pickle_path=data_file
            )
            self.meta, data_label2train_label = transform_pickle_format(pickle_data=data)
            self.is_loaded = True

        else:
            with open(data_file, 'r') as f:
                self.meta = json.load(f)

            if "cifar" in data_file.lower():
                self.meta = load_all_data(self.meta)
                self.is_loaded = True

            else:
                self.is_loaded = False

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            # print(y, x.shape)
            self.sub_meta[y].append(x)
        # sdfa
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl],
                cl,
                transform=transform,
                normal_transform=normal_transform,
                support_aug=support_aug,
                support_aug_num=support_aug_num
            )
            self.sub_dataloader.append(torch.utils.data.DataLoader(
                sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, normal_transform=None, transform=transforms.ToTensor(), target_transform=identity, support_aug=False, support_aug_num=5):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.normal_transform = normal_transform

        self.target_transform = target_transform

        self.support_aug = support_aug
        self.support_aug_num = support_aug_num

    def __getitem__(self, i):
        #print( '%d -%d' %(self.cl,i))
        image_info = self.sub_meta[i]
        if isinstance(image_info, (str, list)):
            image_path = os.path.join(image_info)
            img_original = Image.open(image_path).convert('RGB')

        else:
            img_original = Image.fromarray(image_info)

        target = self.target_transform(self.cl)
        if self.support_aug:
            data = {}
            for i in range(1, self.support_aug_num):
                img = self.transform(img_original)
                data["image_{}".format(i)] = img

            data["image_0"] = self.normal_transform(img_original)
            data["label"] = target
            return data

        else:
            img = self.transform(img_original)
            return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
