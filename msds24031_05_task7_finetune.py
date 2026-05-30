from msds24031_05_task6_linear_probe import get_classification_loaders,evaluate

from msds24031_05_task4_simclr import resnet_encoder

import torch.nn as nn

from pathlib import Path
import matplotlib.pyplot as plt

import torch
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

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = (get_classification_loaders())

    checkpoint = torch.load('models/simclr_model.pth', map_location=device)

    encoder = resnet_encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    model = FineTuneModel(encoder).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    validation_accuracies = train_finetune(model,train_loader,val_loader,optimizer,criterion,device,epochs=20)

    test_accuracy = evaluate(model,test_loader,device)

    print(f'\nfine tuned test accuracy: {test_accuracy:.3f}%')

    save_accuracy_plot(validation_accuracies,'graphs/finetuning_accuracy.png')

    torch.save({'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict()},
                'models/finetuned_encoder.pth')
