from msds24031_05_task4_simclr import resnet_encoder, ProjectionHead, similarity_matrix, nt_xent_loss

from msds24031_05_task3_similarity import simclr_loader

from utils.seed import set_seed

set_seed(2026)

import torch
import torch.nn as nn

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def train_simclr(encoder, projection_head, loader, optimizer, device, epochs = 50):

    losses = []

    encoder.train()
    projection_head.train()

    for epoch in range(epochs):

        total_loss = 0.0

        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')

        for view1, view2, _ in progress_bar:

            view1 = view1.to(device)
            view2 = view2.to(device)

            features1 = encoder(view1)
            features2 = encoder(view2)

            z1 = projection_head(features1)
            z2 = projection_head(features2)

            z = torch.cat([z1, z2], dim=0)

            sim_matrix = similarity_matrix(z)

            loss = nt_xent_loss(sim_matrix, batch_size = view1.size(0), tau = 0.5)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=f'{loss.item():.3f}')

        average_loss = total_loss / len(loader)

        losses.append(average_loss)

        torch.save({'encoder': encoder.state_dict(),
                    'projection_head': projection_head.state_dict(),
                    'losses': losses,
                    'epoch': epoch + 1},
                    'models/simclr_model_latest.pth')
        
        print(f'\nepoch {epoch+1}/{epochs}')
        print(f'average loss : {average_loss:.3f}')

        print('\ncheckpoint saved...')

    return losses

if __name__ == '__main__':

    encoder = resnet_encoder().to(device)
    projection_head = ProjectionHead().to(device)

    loader = simclr_loader(batch_size = 64)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr = 3e-4)

    losses = train_simclr(encoder, projection_head, loader, optimizer, device, epochs = 50)

    torch.save({'encoder': encoder.state_dict(),
                'projection_head': projection_head.state_dict(),
                'losses': losses},
                'models/simclr_model.pth')

    print('\ntraining completed!')

    print('\nsimclr model is saved')

