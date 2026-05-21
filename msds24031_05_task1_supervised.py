from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

set_seed(2026)

train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',train=True)

print('training samples : ', len(train_dataset))