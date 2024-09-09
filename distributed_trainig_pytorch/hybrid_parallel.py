import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Assume we have 4 GPUs available (GPUs 0-3)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

# Data Parallelism - Dataset partitioned across multiple GPUs
def get_dataloader(batch_size):
    # Here, data should be split across different GPUs
    # Simulating with dummy data
    dataset = torch.randn(1000, 3, 224, 224)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

# Define a large model (e.g., ResNet) to be split using Model Parallelism
class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        # Split layers across GPUs
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ).to(0)  # GPU 0
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ).to(1)  # GPU 1
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ).to(2)  # GPU 2
        
        self.fc = nn.Linear(256 * 6 * 6, 1000).to(3)  # GPU 3

    def forward(self, x):
        x = x.to(0)  # Move input to GPU 0
        x = self.layer1(x)
        x = x.to(1)
        x = self.layer2(x)
        x = x.to(2)
        x = self.layer3(x)
        x = x.to(3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model, move it to DDP
model = LargeModel()
model = DDP(model, device_ids=[local_rank])

# Pipeline Parallelism - Assume the model is broken into stages
def forward_pipeline(input_batch):
    # Stage 1 (GPU 0): Input to Layer1
    x = input_batch.to(0)
    x = model.module.layer1(x)
    
    # Stage 2 (GPU 1): Layer1 output to Layer2
    x = x.to(1)
    x = model.module.layer2(x)
    
    # Stage 3 (GPU 2): Layer2 output to Layer3
    x = x.to(2)
    x = model.module.layer3(x)
    
    # Stage 4 (GPU 3): Layer3 output to FC
    x = x.to(3)
    x = x.view(x.size(0), -1)
    output = model.module.fc(x)
    return output

# Training loop
def train():
    dataloader = get_dataloader(batch_size=32)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = forward_pipeline(batch)
            targets = torch.randint(0, 1000, (32,)).to(3)  # Dummy targets on GPU 3
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
