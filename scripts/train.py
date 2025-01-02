import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch

from scripts.model import LogisticRegressionModel

def prepare_mnist(train_bsize = 64, test_bsize = 1000, remove = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    if remove:
        new_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i != remove])
        train_dataset = new_dataset
        
    train_loader = DataLoader(train_dataset, batch_size=train_bsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_bsize, shuffle=False)

    return train_loader, test_dataset

def prepare_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def train_procedure(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return 

def test_single(model, x, y):
    y_pred = model.forward(x)
    return -torch.log(y_pred[0][y])
    
def train(remove = None, epoch = 1):
    model = LogisticRegressionModel(28 * 28, 10)
    train_loader, _ = prepare_mnist(remove = remove)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.008)
    
    name = "trained_without_" + str(remove)
    train_procedure(model, train_loader, criterion, optimizer, num_epochs = epoch)
    torch.save(model.state_dict(), name + ".pth")

    return model
