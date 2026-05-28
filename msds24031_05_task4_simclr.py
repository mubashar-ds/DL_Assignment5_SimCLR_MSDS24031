import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from msds24031_05_task3_similarity import simclr_loader

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# for coisne similarity computation...

def feature_batch(z1, z2):

    return torch.cat([z1, z2], dim =0)

def similarity_matrix(z):

    z = F.normalize(z, dim = 1)

    similarity_matrix = torch.matmul(z, z.T)

    return similarity_matrix

def similarity_heatmap(similarity_matrix, out_path):

    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = similarity_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.colorbar()

    plt.title('Similarity Matrix Before Training')

    plt.tight_layout()

    plt.savefig(out_path, dpi = 300)

    plt.close()

# nt cent contastive loss...

def nt_xent_loss(similarity_matrix, batch_size, tau= 0.5):

    num_views = 2*batch_size

    exp_sim = torch.exp(similarity_matrix/tau)
    mask = ~torch.eye(num_views, dtype=torch.bool,device= similarity_matrix.device)
    exp_sim = exp_sim*mask

    losses = []

    for i in range(num_views):

        pos_index = get_positive_pair(i, batch_size)

        numerator = exp_sim[i, pos_index]
        denominator = exp_sim[i].sum()

        loss = -torch.log(numerator/denominator)
        losses.append(loss)

    losses = torch.stack(losses)

    return losses.mean()

if __name__ == '__main__':

    encoder = resnet_encoder()

    projection_head = ProjectionHead()

    loader = simclr_loader(batch_size=64)

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

    # for cosine similarity computation..

    small_loader = simclr_loader(batch_size=4)

    view1, view2, _ = next(iter(small_loader))

    features1 = encoder(view1)
    features2 = encoder(view2)

    z1 = projection_head(features1)
    z2 = projection_head(features2)

    z = feature_batch(z1, z2)

    sim_matrix = similarity_matrix(z)

    print('\nsimilarity matrix shape :' , sim_matrix.shape)

    similarity_heatmap(sim_matrix, 'results/similarity_matrix_before_training.png')

    # nt xent loss computation..

    loss = nt_xent_loss(sim_matrix, batch_size=4, tau=0.5)

    print('\nnt-xent loss :', round(loss.item(), 4))
