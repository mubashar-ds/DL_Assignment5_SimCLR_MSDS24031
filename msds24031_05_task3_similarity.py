from utils.dataset_splits import get_cifar10_subset, TwoViewDataset

from msds24031_05_task2_augmentations import TwoViewTransform

from msds24031_05_task2_augmentations import simclr_transform

from torch.utils.data import DataLoader

from utils.seed import set_seed

set_seed(2026)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

base_dataset = get_cifar10_subset(data_root = 'data', split_file='splits/train_ssl_unlabeled.txt', train = True)

simclr_dataset = TwoViewDataset(base_dataset, TwoViewTransform(simclr_transform))

loader = DataLoader(simclr_dataset, batch_size = 64, shuffle=False)

view1, view2, _ = next(iter(loader))

print(view1.shape)
print(view2.shape)

def random_encoder():

    encoder = torchvision.models.resnet18(weights=None)

    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    encoder.maxpool = nn.Identity()

    encoder.fc = nn.Identity()

    return encoder

encoder = random_encoder()

with torch.no_grad():

    f1 = encoder(view1)
    f2 = encoder(view2)

print(f1.shape)
print(f2.shape)