import torch
from pathlib import Path

from msds24031_05_task4_simclr import resnet_encoder, ProjectionHead, similarity_matrix

from msds24031_05_task3_similarity import simclr_loader

import matplotlib.pyplot as plt

from utils.seed import set_seed

set_seed(2026)

def save_loss_plot(losses, out_path):

    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))

    plt.plot(losses)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SimCLR Pretraining Loss')

    plt.tight_layout()

    plt.savefig(out_path, dpi=300)

    plt.close()

def similarity_heatmap(similarity_matrix, out_path):

    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = similarity_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(7, 6))

    plt.imshow(matrix)

    plt.colorbar()

    plt.title('Similarity Matrix After Training')

    plt.tight_layout()

    plt.savefig(out_path, dpi = 300)

    plt.close()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('models/simclr_model.pth', map_location =device)

    encoder = resnet_encoder().to(device)
    projection_head = ProjectionHead().to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    projection_head.load_state_dict(checkpoint['projection_head'])

    losses = checkpoint['losses']
    save_loss_plot(losses, 'graphs/simclr_pretraining_loss.png')

    loader = simclr_loader(batch_size = 4)
    view1, view2, _ = next(iter(loader))

    encoder.eval()
    projection_head.eval()

    with torch.no_grad():
            
        view1 = view1.to(device)
        view2 = view2.to(device)

        features1 = encoder(view1)
        features2 = encoder(view2)

        z1 = projection_head(features1)
        z2 = projection_head(features2)

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = similarity_matrix(z)

    similarity_heatmap(sim_matrix, 'results/similarity_matrix_after_training.png')
