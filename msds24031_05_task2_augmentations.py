import torchvision.transforms as T

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.visualization import save_augmentation_grid

set_seed(2026)

simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
std=(0.2470, 0.2435, 0.2616))
])

class TwoViewTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):

        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2
    
dataset = get_cifar10_subset(data_root='data',split_file = 'splits/train_labeled_10percent.txt', train=True)

two_view_transform = TwoViewTransform(simclr_transform)

originals = []
view1s = []
view2s = []

for i in range(10):

    image, label = dataset[i]

    view1, view2 = two_view_transform(image)

    originals.append(image)

    view1s.append(view1)

    view2s.append(view2)

save_augmentation_grid(originals,view1s,view2s, 'results/augmentation_examples.png')

print('saved augmentation examples.')

