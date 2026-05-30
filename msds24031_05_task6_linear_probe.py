from torch.utils.data import DataLoader
from utils.dataset_splits import get_cifar10_subset

import torch
import torchvision.transforms as T

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

        print(f'\nepoch {epoch+1}/{epochs}')
        print(f'loss: {average_loss:.3f}')
        print(f'validation accuracy : {validation_accuracy:.3f}%')

def run_experiment(encoder, train_loader, val_loader, test_loader, device):

    freeze_encoder(encoder)
    model = LinearProbe(encoder).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=3e-4)
    train_probe(model,train_loader,val_loader,optimizer,criterion,device,epochs=20)
    test_accuracy = evaluate(model,test_loader,device)

    return test_accuracy

def save_accuracy_plot(random_accuracy,simclr_accuracy,out_path):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True,exist_ok =True)

    plt.figure(figsize=(5,4))

    plt.bar(['Random', 'SimCLR'],[random_accuracy, simclr_accuracy])
    plt.ylabel('Test Accuracy (%)')
    plt.title('Linear Probe Evaluation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = (get_classification_loaders())

    print('\nrandom encoder linear probe...')

    random_encoder = resnet_encoder()
    random_accuracy = run_experiment(random_encoder,train_loader,val_loader,test_loader,device)

    print(f'\nrandom probe test accuracy: {random_accuracy:.3f}%')

    checkpoint = torch.load('models/simclr_model.pth',map_location=device)

    simclr_encoder = resnet_encoder()
    simclr_encoder.load_state_dict(checkpoint['encoder'])

    print('\nsimclr encoder linear probe..')

    simclr_accuracy = run_experiment(simclr_encoder,train_loader,val_loader,test_loader,device)

    print(f'\nsimclr probe test accuracy: {simclr_accuracy:.3f}%')

    save_accuracy_plot(random_accuracy,simclr_accuracy,'graphs/linear_probe_accuracy.png')