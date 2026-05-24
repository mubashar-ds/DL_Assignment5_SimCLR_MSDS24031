import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from msds24031_05_task3_similarity import simclr_loader

def resnet_encoder():

    encoder = torchvision.models.resnet18(weights=None)

    encoder.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)

    encoder.maxpool = nn.Identity()

    encoder.fc = nn.Identity()

    return encoder

class ProjectionHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):

        return self.layers(x)
    
def get_positive_pair(index, batch_size):

    if index < batch_size:
        return index + batch_size

    return index - batch_size

if __name__ == '__main__':

    encoder = resnet_encoder()

    projection_head = ProjectionHead()

    loader = simclr_loader()

    view1, view2, _ = next(iter(loader))

    features = encoder(view1)

    print('\nfeature shape :', features.shape)

    z = projection_head(features)

    print('\nprojection shape : ', z.shape)

    batch_size = 4

    print('\npositive pairs\n')

    for i in range(batch_size):

        print(
            f'Image {i} : '
            f'View 1 Index = {i} , '
            f'View 2 Index = {get_positive_pair(i, batch_size)}'
        )




   