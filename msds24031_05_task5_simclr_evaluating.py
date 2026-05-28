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

def feature_similarities(encoder,projection_head,loader, device, use_projection_head=False):

    encoder.eval()
    projection_head.eval()

    view1, view2, _ = next(iter(loader))

    with torch.no_grad():

        view1 = view1.to(device)
        view2 = view2.to(device)

        features1 = encoder(view1)
        features2 = encoder(view2)

        if use_projection_head:
            z1 = projection_head(features1)
            z2 = projection_head(features2)

        else:
            z1 = features1
            z2 = features2

        z = torch.cat([z1, z2], dim=0)

        sim_matrix = similarity_matrix(z)

    same_similarity = []
    different_similarity = []

    batch_size = view1.size(0)

    for i in range(batch_size):

        positive_index = i + batch_size
        same_similarity.append(sim_matrix[i, positive_index].item())

    num_views = 2*batch_size

    for i in range(num_views):

        for j in range(num_views):

            if i == j:
                continue

            positive_index = (i + batch_size if i < batch_size else i - batch_size)

            if j == positive_index:
                continue

            different_similarity.append(sim_matrix[i, j].item())

    average_same = (sum(same_similarity) / len(same_similarity))
    average_different = (sum(different_similarity) / len(different_similarity))

    return (average_same, average_different,sim_matrix)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = simclr_loader(batch_size=4)

    untrained_encoder = resnet_encoder().to(device)
    untrained_projection_head = ProjectionHead().to(device)

    before_same, before_different, _ = (feature_similarities(untrained_encoder, untrained_projection_head,
                                                             loader, device,use_projection_head=False))

    checkpoint = torch.load('models/simclr_model.pth', map_location =device)

    encoder = resnet_encoder().to(device)
    projection_head = ProjectionHead().to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    projection_head.load_state_dict(checkpoint['projection_head'])
    
    after_same, after_different, sim_matrix = (feature_similarities(encoder, projection_head,
                                                                    loader, device))

    losses = checkpoint['losses']
    save_loss_plot(losses, 'graphs/simclr_pretraining_loss.png')
    
    similarity_heatmap(sim_matrix, 'results/similarity_matrix_after_training.png')

    print('\nfeature similarity comparison')

    print(f'\nsame image views : ')
    print(f'before = {before_same:.3f}')
    print(f'after = {after_same:.3f}')

    print(f'\ndifferent image views : ')
    print(f'before = {before_different:.3f}')
    print(f'after = {after_different:.3f}')
