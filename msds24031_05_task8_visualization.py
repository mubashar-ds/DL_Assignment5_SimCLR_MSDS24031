import torch

import numpy as np

from utils.seed import set_seed
from utils.visualization import save_2d_feature_plot

from msds24031_05_task6_linear_probe import get_classification_loaders

from msds24031_05_task4_simclr import resnet_encoder

set_seed(2026)

def extract_features(encoder,loader,device, max_samples=1000):

    encoder.eval()
    features_list = []
    labels_list = []
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
           
            features = encoder(images)
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

            total += images.size(0)

            if total >= max_samples:
                break

    features = torch.cat(features_list, dim=0)[:max_samples]
    labels = torch.cat(labels_list, dim=0)[:max_samples]

    return (features.numpy(), labels.numpy())

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader, _ = (get_classification_loaders())

    # random encoder pca..

    random_encoder = (resnet_encoder().to(device))
    random_features, random_labels = (extract_features(random_encoder,val_loader,device))

    save_2d_feature_plot(random_features,random_labels,'results/random_encoder_pca.png',method='pca',title='Random Encoder PCA')

    # simclr encoder pca..

    simclr_checkpoint = torch.load('models/simclr_model.pth',map_location=device)

    simclr_encoder = (resnet_encoder().to(device))
    simclr_encoder.load_state_dict(simclr_checkpoint['encoder'])
    simclr_features, simclr_labels = (extract_features(simclr_encoder,val_loader,device))

    save_2d_feature_plot(simclr_features,simclr_labels,'results/simclr_encoder_pca.png',method='pca',title='SimCLR Encoder PCA')

    # fine tuned encoder pca....

    finetuned_checkpoint = torch.load('models/finetuned_encoder.pth',map_location=device)

    finetuned_encoder = (resnet_encoder().to(device))
    finetuned_encoder.load_state_dict(finetuned_checkpoint['encoder'])
    finetuned_features, finetuned_labels = (extract_features(finetuned_encoder,val_loader,device))

    save_2d_feature_plot(finetuned_features,finetuned_labels,'results/finetuned_encoder_pca.png',method='pca',
                         title='Fine-Tuned Encoder PCA')
