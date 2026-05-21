from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

set_seed(2026)

transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.4914, 0.4822, 0.4465),
std=(0.2470, 0.2435, 0.2616))])
                          
train_transform = transform
test_transform = transform

train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',
                                   train=True, transform= train_transform)

test_dataset = get_cifar10_subset(data_root ='data', split_file='splits/test.txt',
                                   train=False, transform = test_transform)

val_dataset = get_cifar10_subset(data_root ='data', split_file='splits/val.txt',
                                   train=False, transform = test_transform)

print('training samples : ', len(train_dataset))
print('test samples : ', len(test_dataset))
print('validation samples : ', len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle= False)

val_loader = DataLoader(val_dataset, batch_size= 64, shuffle =False)

images, labels = next(iter(train_loader))

print('images shape :', images.shape)
print('labels shape : ', labels.shape)
print('labels dtype : ', labels.dtype)