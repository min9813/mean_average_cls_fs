# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class AugDataset(SimpleDataset):

    def __init__(self, data_file, transform, target_transform=identity, aug_num=5):
        super(AugDataset, self).__init__(data_file=data_file, transform=transform, target_transform=target_transform)
        self.aug_num = aug_num

    def __getitem__(self, i):
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
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(
                self.sub_meta[cl], 
                cl, 
                transform = transform,
                normal_transform=normal_transform,
                support_aug=support_aug, 
                support_aug_num=support_aug_num
                )
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
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

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img_original = Image.open(image_path).convert('RGB')

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
