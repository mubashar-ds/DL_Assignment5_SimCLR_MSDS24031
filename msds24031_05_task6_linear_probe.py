from torch.utils.data import DataLoader

from utils.dataset_splits import get_cifar10_subset

import torch
import torchvision.transforms as T

import torchvision
import torch.nn as nn

from msds24031_05_task4_simclr import resnet_encoder

from pathlib import Path
import matplotlib.pyplot as plt

def get_classification_loaders(batch_size=64):

    transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616))])

    train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',
                                   train=True, transform= transform)

    test_dataset = get_cifar10_subset(data_root ='data', split_file='splits/test.txt',
                                    train=False, transform = transform)

    val_dataset = get_cifar10_subset(data_root ='data', split_file='splits/val.txt',
                                    train=True, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle= False)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle =False)

    return train_loader, val_loader, test_loader

class LinearProbe(nn.Module):

    def __init__(self,encoder):
        super().__init__()

        self.encoder = encoder
        self.classifier = nn.Linear(512, 10)

    def forward(self,x):
        features = self.encoder(x)
        outputs = self.classifier(features)

        return outputs
    
def freeze_encoder(encoder):
    for param in encoder.parameters():
        param.requires_grad=False

def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions =outputs.argmax(dim =1)
            correct += (predictions==labels).sum().item()

            total += labels.size(0)

    return 100*correct/total

def train_probe(model,train_loader,val_loader,optimizer, criterion,device,epochs=20):

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss/len(train_loader)

        validation_accuracy = evaluate(model, val_loader,device)

        print(f'epoch {epoch+1}/{epochs}')
        print(f'loss: {average_loss:.3f}')
        print(f'validation accuracy : {validation_accuracy:.3f}%')

if __name__ == '__main__':

    encoder = resnet_encoder()
    freeze_encoder(encoder)

    model = LinearProbe(encoder)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader,test_loader, val_loader = (get_classification_loaders())

    encoder = resnet_encoder()
    freeze_encoder(encoder)
    model = LinearProbe(encoder).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=3e-4)

    train_probe(model,train_loader,val_loader,optimizer,criterion,device,epochs=1)
