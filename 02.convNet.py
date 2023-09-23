import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Load Datasets
trainset = datasets.FashionMNIST(root="./datasets/", train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.FashionMNIST(root="./datasets/", train=False, transform=transforms.ToTensor(), download=True)
print(len(trainset), len(testset))

# Create DataLoader Objects
trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=64)
print(len(trainloader), len(testloader))

# Visualize the data
first_batch_images, first_batch_labels = next(iter(trainloader))
print(first_batch_images.shape, first_batch_labels.shape)

for i in range(5):
    rand_idx = torch.randint(0, len(first_batch_images), size=[1]).item()
    image, label = first_batch_images[rand_idx], first_batch_labels[rand_idx]
    plt.imshow(image.squeeze())
    plt.title(label=label.item())
    # plt.show()

# Time to create the NeuralNetwork
class NeuralNet(nn.Module):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_shape, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=output_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create device and model
class_names = trainset.classes
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNet(input_shape=784, output_shape=len(class_names)).to(device=device)

# Create an instance and test it out
# tensor = torch.randint(0, 255, size=[5, 1, 28, 28])
# print(tensor.shape)

# tensor = tensor.reshape(tensor.shape[0], -1).to(device)
# print(tensor.dtype)
# print(model(tensor.view(-1, 28*28)))

# Create loss and optimizer function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

def training(epochs: int,
             trainloader: torch.utils.data.DataLoader,
             model: NeuralNet,
             criterion: torch.nn,
             optimizer: torch.optim):
    """Function to train the Model on DataLoader Object"""
    for i in range(1, epochs + 1):
        train_loss, correct = 0, 0
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device) 
            data = data.reshape(data.shape[0], -1)
            targets = model(data)
            # print(targets.argmax(dim=1), labels)        
            loss = criterion(targets, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            correct += (labels == targets.argmax(dim=1)).sum().item()

        print(f"epoch: {i} | train_loss: {train_loss/len(trainloader): .6f} | train_accuracy: {correct/(64 * len(trainloader)): .6f}") 

training(5, trainloader, model, criterion, optimizer)


# Time to test/evaluate the model
with torch.inference_mode():
    test_loss, correct = 0, 0
    for data, labels in testloader:
        data, label = data.to(device), labels.to(device)
        data = data.reshape(data.shape[0], -1)
        scores = model(data)

        correct += (labels == scores.argmax(dim=1)).sum().item()
        test_loss += criterion(scores, labels)
    
    print(f"test_loss: {test_loss/len(testloader): .6f} | test_accuracy: {correct/(64 * len(testloader)): .6f}")        
