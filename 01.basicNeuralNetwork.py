import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully Connected Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 5

# Load Dataset
trainDataset = datasets.MNIST(root="./datasets/", train=True, transform=transforms.ToTensor(), download=True)
testDataset = datasets.MNIST(root="./datasets/", train=False, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size)

# Initialize Network
model = NeuralNetwork(input_size=input_size, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        data = data.to(device)
        targets = targets.to(device)
        # Convert to correct shape (aka flatten)
        data = data.reshape(data.shape[0], -1)
        # forward
        predictions = model(data)
        loss = criterion(predictions, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Check accuracy on training & test the Model
def check_accuracy(loader: torch.utils.data.DataLoader, model: NeuralNetwork) -> None:
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.inference_mode():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} correct out of {num_samples}")


check_accuracy(testLoader, model)