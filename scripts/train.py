import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch
import os, pickle, math

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Manager

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

    return train_loader, test_loader

def prepare_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def train_procedure(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device = None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
             
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
        test(model, test_loader)
    
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
    
def train(remove = None, epoch = 5, device = None):
    model = LogisticRegressionModel(28 * 28, 10)
    if device:
        model.to(device)
    train_loader, test_loader = prepare_mnist(remove = remove)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3)
    
    name = "data/models/trained_without_" + str(remove)
    train_procedure(model, train_loader, test_loader, criterion, optimizer, num_epochs = epoch, device = device)

    model.to('cpu')
    torch.save(model.state_dict(), name + ".pth")
    print("saved to", name)
    return model

def load_model(idx=None, epoch=20, device=None, rank=None):
    if device == None:
        print("device none", rank, device)
    if os.path.exists("data/models/trained_without_" + str(idx) + ".pth"):
        model = LogisticRegressionModel(28 * 28, 10)
        model.load_state_dict(torch.load("data/models/trained_without_" + str(idx) + ".pth"))
    else:
        model = train(remove=idx, epoch=epoch, device=device)
    return model

def save_result(lst, dir):
    with open(dir, 'wb') as f:
        pickle.dump(lst, f)
    
def leave_one_out(train_idx, test_idx, device=None, rank=None):
    print(device, rank)
    _model = load_model(device=device, rank=rank)
    model = load_model(train_idx, device=device, rank=rank)
     
    _, test_dataset = prepare_mnist_dataset(remove=None)
    x = test_dataset[test_idx[0]][0].view(1, -1)
    y = torch.tensor([test_dataset[test_idx[0]][1]]) 
    return test_single(model, x, y) - test_single(_model, x, y)

def ddp_setup(rank, world_size, backend="nccl"):
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    return device

def worker_fn(rank, world_size, train_indices, test_idx, return_list):
    device = ddp_setup(rank, world_size, backend="nccl")
    print("starting subprocess", device)

    chunk_size = math.ceil(len(train_indices) / world_size)
    start = rank * chunk_size
    end = min(start + chunk_size, len(train_indices))
    local_train_indices = train_indices[start:end]

    print(f"[Rank {rank}] will handle train_indices from {start} to {end-1}.")

    local_results = []
    for train_idx in local_train_indices:
        diff = leave_one_out(train_idx, test_idx, device=device, rank=rank)
        local_results.append((train_idx, diff.detach().numpy()))

    return_list.extend(local_results)
    print(f"[Rank {rank}] returned {len(local_results)} results.")
    dist.destroy_process_group()

def calculate_retrained_loss(train_idx, test_idx):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    world_size = torch.cuda.device_count()
    print("processes", world_size)

    manager = Manager()
    return_list = manager.list()
    mp.spawn(
        worker_fn,
        args=(world_size, train_idx, test_idx, return_list),
        nprocs=world_size,
        join=True
    )
    return_list = list(return_list)
    print(f"Finished with list of length {len(return_list)}")
    
    return return_list