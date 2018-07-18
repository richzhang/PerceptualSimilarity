import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from IPython import embed

class JNDDataset(BaseDataset):
    def initialize(self, dataroot, load_size=64):
        self.root = dataroot
        self.load_size = load_size

        self.dir_p0 = os.path.join(self.root, 'p0')
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = os.path.join(self.root, 'p1')
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Scale(load_size))
        transform_list += [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_S = os.path.join(self.root, 'same')
        self.same_paths = make_dataset(self.dir_S,mode='np')
        self.same_paths = sorted(self.same_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.transform(p1_img_)

        same_path = self.same_paths[index]
        same_img = np.load(same_path).reshape((1,1,1,)) # [0,1]

        same_img = torch.FloatTensor(same_img)

        return {'p0': p0_img, 'p1': p1_img, 'same': same_img,
            'p0_path': p0_path, 'p1_path': p1_path, 'same_path': same_path}

    def __len__(self):
        return len(self.p0_paths)
