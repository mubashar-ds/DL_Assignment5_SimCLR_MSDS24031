from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader

import torchvision
import torch.nn as nn

import matplotlib.pyplot as plt

from utils.metrics import save_confusion_matrix

set_seed(2026)

transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.4914, 0.4822, 0.4465),
std=(0.2470, 0.2435, 0.2616))])
                          
train_transform = transform
test_transform = transform

# splits loader...

train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',
                                   train=True, transform= train_transform)

test_dataset = get_cifar10_subset(data_root ='data', split_file='splits/test.txt',
                                   train=False, transform = test_transform)

val_dataset = get_cifar10_subset(data_root ='data', split_file='splits/val.txt',
                                   train=True, transform = test_transform)

print('training samples : ', len(train_dataset))
print('test samples : ', len(test_dataset))
print('validation samples : ', len(val_dataset))

# dataloades...

train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle= False)

val_loader = DataLoader(val_dataset, batch_size= 64, shuffle =True)

images, labels = next(iter(train_loader))

print('images shape :', images.shape)
print('labels shape : ', labels.shape)
print('labels dtype : ', labels.dtype)

# modified resnet18 for supervised training..

def modified_resnet18():

    model = torchvision.models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1, bias = False)

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, 10)

    return model

model = modified_resnet18()

outputs = model(images)

print('output shape :', outputs.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def train_one_epoch(model,loader,criterion, optimizer, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion,device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_accuracy(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim = 1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    return 100 * correct / total

train_losses = []
val_losses = []
num_epochs = 10

best_val_loss = float('inf')

# model training loop... 

for epoch in range(num_epochs):

    train_loss = train_one_epoch(model,train_loader, criterion, optimizer, device)

    val_loss = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    
    val_losses.append(val_loss)

    val_acc = evaluate_accuracy(model, val_loader, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(model.state_dict(), 'models/supervised_best.pt')

    print(f'\nepoch {epoch+1}/{num_epochs} : ')

    print(f'\ntrain loss : {train_loss :.3f}: ')
    print(f'validation loss : {val_loss :.3f} : ')
    print(f'\nvalidation accuracy : {val_acc :.3f}%')

# loss curve...

plt.figure(figsize = (8,5))

plt.plot(train_losses,label = 'Train Loss')
plt.plot(val_losses,label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.tight_layout()

plt.savefig('graphs/supervised_loss.png')

plt.close()

# test evaluation ...

model.load_state_dict(torch.load('models/supervised_best.pt'))

test_acc = evaluate_accuracy(model,test_loader,device)

print('test accuracy :', test_acc)

# confusion matrix,..

def get_predictions(model, loader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred

model.load_state_dict(torch.load('models/supervised_best.pt', map_location=device))

y_true, y_pred = get_predictions(model, test_loader,device)

save_confusion_matrix(y_true,y_pred,
                      'results/supervised_confusion_matrix.png', 
                      title='Supervised Baseline')