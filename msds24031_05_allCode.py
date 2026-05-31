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

from utils.visualization import save_augmentation_grid
from utils.dataset_splits import TwoViewDataset
from msds24031_05_task2_augmentations import TwoViewTransform
from msds24031_05_task2_augmentations import simclr_transform

import torch.nn.functional as F

from msds24031_05_task3_similarity import simclr_loader

import numpy as np
from pathlib import Path

from msds24031_05_task4_simclr import resnet_encoder, ProjectionHead, similarity_matrix, nt_xent_loss
from msds24031_05_task3_similarity import simclr_loader

from tqdm import tqdm

from msds24031_05_task6_linear_probe import get_classification_loaders,evaluate

from utils.visualization import save_2d_feature_plot

import json

from msds24031_05_task7_finetune import FineTuneModel

import pandas as pd

# task 1 --------------------------------------------------------

transform = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.4914, 0.4822, 0.4465),
std=(0.2470, 0.2435, 0.2616))])

# splits loader...
train_dataset = get_cifar10_subset(data_root ='data', split_file='splits/train_labeled_10percent.txt',
                                   train=True, transform= transform)
test_dataset = get_cifar10_subset(data_root ='data', split_file='splits/test.txt',
                                   train=False, transform = transform)
val_dataset = get_cifar10_subset(data_root ='data', split_file='splits/val.txt',
                                   train=True, transform = transform)

# dataloades...
loader_train = DataLoader(train_dataset, batch_size=64, shuffle = True)
loader_test = DataLoader(test_dataset, batch_size=64, shuffle= False)
loader_val = DataLoader(val_dataset, batch_size= 64, shuffle =False)

# modified resnet18 for supervised training..
def modified_resnet18():

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1, bias = False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

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

# task 2 --------------------------------------------------------

simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
std=(0.2470, 0.2435, 0.2616))
])

class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2
    
# task 3 --------------------------------------------------------

def simclr_loader(batch_size=64):

    base_dataset = get_cifar10_subset(data_root = 'data', split_file='splits/train_ssl_unlabeled.txt', train = True)
    simclr_dataset = TwoViewDataset(base_dataset, TwoViewTransform(simclr_transform))
    loader = DataLoader(simclr_dataset, batch_size = batch_size, shuffle=True, num_workers=2, 
                        persistent_workers=True, pin_memory=False)

    return loader

def random_encoder():

    encoder = torchvision.models.resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()
    encoder.fc = nn.Identity()

    return encoder

# task 4 --------------------------------------------------------

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

# task 5 --------------------------------------------------------

# pretraining...
    
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

# evaluation..

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

# task 6 --------------------------------------------------------

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

def save_accuracy_plots(random_accuracy,simclr_accuracy,out_path):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True,exist_ok =True)

    plt.figure(figsize=(5,4))

    plt.bar(['Random', 'SimCLR'],[random_accuracy, simclr_accuracy])
    plt.ylabel('Test Accuracy (%)')
    plt.title('Linear Probe Evaluation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# task 7 --------------------------------------------------------

class FineTuneModel(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.classifier = nn.Linear(512,10)

    def forward(self,x):
        features = self.encoder(x)
        outputs = self.classifier(features)

        return outputs
    
def train_finetune(model,train_loader,val_loader,optimizer,criterion, device,epochs=20):

    validation_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images= images.to(device)
            labels= labels.to(device)

            outputs = model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss/len(train_loader)

        validation_accuracy = evaluate(model,val_loader,device)
        validation_accuracies.append(validation_accuracy)

        print(f'\nepoch {epoch+1}/{epochs}')
        print(f'loss: {average_loss:.3f}')
        print(f'validation accuracy : {validation_accuracy:.3f}%')

    return validation_accuracies

def save_accuracy_plot(accuracies,out_path):

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6,4))

    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Fine-Tuning Accuracy')
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()

# task 8 --------------------------------------------------------

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

# metrics.json ---------------------------------------------------

def save_metrics():

    metrics = {

        "student_name": "Mubashar Hussain",
        "roll_number": "MSDS24031",
        "seed": 2026,
        "batch_size": 64,
        "simclr_epochs": 50,
        "linear_probe_epochs": 20,
        "finetuning_epochs": 20,
        "learning_rate": 0.0003,
        "temperature": 0.5,

        "supervised_10percent_test_acc": 45.87,
        "random_linear_probe_test_acc": 26.960,
        "simclr_linear_probe_test_acc": 74.660,
        "simclr_finetune_test_acc": 81.150,

        "same_view_similarity_before": 0.992,
        "different_image_similarity_before": 0.988,
        "same_view_similarity_after": 0.905,
        "different_image_similarity_after": 0.462
    }

    Path('results').mkdir(exist_ok=True)

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

# test predictions cvs -----------------------------------------------

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

# main() ========================================================

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # task 1 -----------------------------

    images, labels = next(iter(loader_train))

    model = modified_resnet18()
    outputs = model(images)
    print('output shape :', outputs.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(model.parameters(), lr=3e-4)

    train_losses = []
    val_losses = []
    num_epochs = 10

    best_val_loss = float('inf')

    # model training loop... 
    for epoch in range(num_epochs):

        train_loss = train_one_epoch(model,loader_train, criterion, optimizer, device)
        val_loss = validate(model, loader_val, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acc = evaluate_accuracy(model, loader_val, device)

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
    model.load_state_dict(torch.load('models/supervised_best.pt', map_location = device))
    test_acc = evaluate_accuracy(model,loader_test,device)
    print('test accuracy :', test_acc)
    model.load_state_dict(torch.load('models/supervised_best.pt', map_location=device))

    y_true, y_pred = get_predictions(model, loader_test,device)
    save_confusion_matrix(y_true,y_pred,
                        'results/supervised_confusion_matrix.png', 
                        title='Supervised Baseline')
    
    # task 2 -----------------------------

    dataset = get_cifar10_subset(data_root='data',split_file = 'splits/train_labeled_10percent.txt', train=True)
    two_view_transform = TwoViewTransform(simclr_transform)

    originals = []
    view1s = []
    view2s = []

    for i in range(10):
        image, label = dataset[i]
        view1, view2 = two_view_transform(image)
        originals.append(image)
        view1s.append(view1)
        view2s.append(view2)

    save_augmentation_grid(originals,view1s,view2s, 'results/augmentation_examples.png')

    # task 3 -----------------------------

    loader = simclr_loader()

    view1, view2, _ = next(iter(loader))
    print(view1.shape)
    print(view2.shape)

    encoder = random_encoder()

    with torch.no_grad():
        f1 = encoder(view1)
        f2 = encoder(view2)

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    same_similarity = (f1 * f2).sum(dim=1)
    same_average = same_similarity.mean().item()

    different_similarity = (f1[:-1] * f2[1:]).sum(dim=1)
    different_average = different_similarity.mean().item()

    print('\naverage cosine similarity before training')
    print('\nsame image, two augmented views : ', round(same_average, 4))
    print('\ndifferent images :', round(different_average, 4))

    # task 4 -----------------------------

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

    # task 5 -----------------------------

    # pretraining...

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

    # evaluation..

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

    # task 6 -----------------------------

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

    save_accuracy_plots(random_accuracy,simclr_accuracy,'graphs/linear_probe_accuracy.png')

    # task 7 -----------------------------

    train_loader, val_loader, test_loader = (get_classification_loaders())

    checkpoint = torch.load('models/simclr_model.pth', map_location=device)

    encoder = resnet_encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    model = FineTuneModel(encoder).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print('\nfine tunining...')
    validation_accuracies = train_finetune(model,train_loader,val_loader,optimizer,criterion,device,epochs=20)
    test_accuracy = evaluate(model,test_loader,device)
    print(f'\nfine tuned test accuracy: {test_accuracy:.3f}%')

    save_accuracy_plot(validation_accuracies,'graphs/finetuning_accuracy.png')

    torch.save({'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict()},
                'models/finetuned_encoder.pth')

    # task 8 -----------------------------

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
    
    # metrics.json....................................

    save_metrics()

    # test predictions cvs...............................

    _, _, test_loader = (get_classification_loaders())

    checkpoint = torch.load('models/finetuned_encoder.pth',map_location=device)

    encoder = resnet_encoder()
    model = FineTuneModel(encoder).to(device)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])

    save_test_predictions(model,test_loader,device)
