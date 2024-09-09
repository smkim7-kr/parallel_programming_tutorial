import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset
import os
import time

# Example dataset
class ExampleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3, 224, 224)  # Random data
        self.labels = torch.randint(0, 10, (size,))  # Random labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, 10)  # Assuming input size 224x224

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc1(x)
        return x

# DataParallel Approach
def train_dataparallel(model, dataloader, device):
    model = DataParallel(model)  # Wrap model with DataParallel
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    
    for epoch in range(2):  # Train for 2 epochs for quick comparison
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[DataParallel] Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

    end_time = time.time()
    print(f"[DataParallel] Training Time: {end_time - start_time:.2f} seconds\n")

# FSDP Approach
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_fsdp(rank, world_size, dataloader):
    setup(rank, world_size)

    model = SimpleCNN().to(rank)
    model = FSDP(model)  # Wrap model with FSDP

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    
    for epoch in range(2):  # Train for 2 epochs for quick comparison
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[FSDP] Rank {rank}, Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

    end_time = time.time()
    print(f"[FSDP] Rank {rank} Training Time: {end_time - start_time:.2f} seconds\n")

    cleanup()

if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = ExampleDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train with DataParallel
    model_dp = SimpleCNN()
    train_dataparallel(model_dp, dataloader, device)

    # Train with FSDP
    world_size = torch.cuda.device_count()
    mp.spawn(train_fsdp, args=(world_size, dataloader), nprocs=world_size, join=True)
