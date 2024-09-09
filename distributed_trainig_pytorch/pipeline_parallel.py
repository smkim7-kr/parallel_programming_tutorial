import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# Example dataset
class ExampleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3, 224, 224)  # Random data
        self.labels = torch.randint(0, 10, (size,))  # Random labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example model split across two GPUs
class PipelineParallelModel(nn.Module):
    def __init__(self, device0, device1):
        super(PipelineParallelModel, self).__init__()
        self.device0 = device0
        self.device1 = device1
        
        # Layer 1 and 2 will be on device0
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(self.device0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device0)
        
        # Layer 3 and 4 will be on device1
        self.fc1 = nn.Linear(16 * 112 * 112, 512).to(self.device1)
        self.fc2 = nn.Linear(512, 10).to(self.device1)

    def forward(self, x):
        # Forward pass on device0
        x = self.pool(torch.relu(self.conv1(x.to(self.device0))))
        x = x.view(-1, 16 * 112 * 112).to(self.device1)
        
        # Forward pass on device1
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_pipeline_parallel(rank, world_size):
    # Setup distributed environment
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set devices
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    # Create the model and move to corresponding devices
    model = PipelineParallelModel(device0, device1)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[device0, device1])
    
    # Example dataset and dataloader
    dataset = ExampleDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(2):  # Train for 2 epochs for demonstration
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(device1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train_pipeline_parallel, args=(world_size,), nprocs=world_size, join=True)
