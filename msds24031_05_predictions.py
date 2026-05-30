from msds24031_05_task6_linear_probe import get_classification_loaders
from msds24031_05_task4_simclr import resnet_encoder
from msds24031_05_task7_finetune import FineTuneModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

def save_test_predictions(model,test_loader,device):

    model.eval()
    rows = []
    image_index = 0

    with torch.no_grad():
        for images,labels in test_loader:
            images =images.to(device)
            logits=model(images)

            probabilities = F.softmax(logits,dim=1)
            predictions = logits.argmax(dim=1)

            probabilities = probabilities.cpu()
            predictions = predictions.cpu()

            for i in range(len(labels)):
                row = {
                    'image_index':
                    image_index,

                    'true_label':
                    labels[i].item(),

                    'predicted_label':
                    predictions[i].item()}

                for c in range(10):
                    row[f'prob_class_{c}'] =probabilities[i,c].item()

                rows.append(row)
                image_index += 1

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv('results/test_predictions.csv',index=False)

if __name__ == '__main__':

    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('results/test_predictions.csv')

    print(df.shape)

    prob_columns = [f'prob_class_{i}' for i in range(10)]
    print(df.loc[0, prob_columns].sum())

    _, _, test_loader = (get_classification_loaders())

    checkpoint = torch.load('models/finetuned_encoder.pth',map_location=device)

    encoder = resnet_encoder()
    model = FineTuneModel(encoder).to(device)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])

    save_test_predictions(model,test_loader,device)
