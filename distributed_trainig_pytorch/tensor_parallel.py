import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

# Example model with tensor parallelism
class TensorParallelModel(nn.Module):
    def __init__(self, device0, device1):
        super(TensorParallelModel, self).__init__()
        self.device0 = device0
        self.device1 = device1
        
        # Split the linear layer's weights across devices
        self.fc1_weight_part0 = nn.Parameter(torch.randn(16 * 112 * 112, 256).to(self.device0))
        self.fc1_weight_part1 = nn.Parameter(torch.randn(16 * 112 * 112, 256).to(self.device1))
        
        self.fc2 = nn.Linear(512, 10).to(self.device1)  # Fully connected layer on device1

    def forward(self, x):
        # Simulate a convolutional operation (for simplicity)
        x = x.view(-1, 16 * 112 * 112)
        
        # Perform tensor parallelism on the first linear layer
        x_part0 = torch.matmul(x.to(self.device0), self.fc1_weight_part0)
        x_part1 = torch.matmul(x.to(self.device1), self.fc1_weight_part1)
        
        # Concatenate the results and pass to the next layer
        x = torch.cat((x_part0, x_part1), dim=1).to(self.device1)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_tensor_parallel(rank, world_size):
    setup(rank, world_size)

    # Set devices
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
    
    # Create the model and move to corresponding devices
    model = TensorParallelModel(device0, device1)
    
    # Example dataset and dataloader
    dataset = ExampleDataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device1)  # Loss computed on device1
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
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train_tensor_parallel, args=(world_size,), nprocs=world_size, join=True)
