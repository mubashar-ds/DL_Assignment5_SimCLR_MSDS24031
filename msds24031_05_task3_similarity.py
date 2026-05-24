from utils.dataset_splits import get_cifar10_subset, TwoViewDataset

from msds24031_05_task2_augmentations import TwoViewTransform

from msds24031_05_task2_augmentations import simclr_transform

from torch.utils.data import DataLoader

from utils.seed import set_seed

set_seed(2026)

base_dataset = get_cifar10_subset(data_root = 'data', split_file='splits/train_ssl_unlabeled.txt', train = True)

simclr_dataset = TwoViewDataset(base_dataset, TwoViewTransform(simclr_transform))

loader = DataLoader(simclr_dataset, batch_size = 64, shuffle=False)

view1, view2, _ = next(iter(loader))

print(view1.shape)
print(view2.shape)