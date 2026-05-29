from torch.utils.data import DataLoader

from utils.dataset_splits import get_cifar10_subset

import torch
import torchvision.transforms as T

def get_classification_loaders(batch_size=64):

    transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616))])

    train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',
                                   train=True, transform= transform)

    test_dataset = get_cifar10_subset(data_root ='data', split_file='splits/test.txt',
                                    train=False, transform = transform)

    val_dataset = get_cifar10_subset(data_root ='data', split_file='splits/val.txt',
                                    train=True, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle= False)

    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle =False)

    return train_loader, val_loader, test_loader