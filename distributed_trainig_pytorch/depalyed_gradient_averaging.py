import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Assume we have 4 GPUs available
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Example model (simple feedforward network)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, move to correct device
model = SimpleModel().to(local_rank)

# Wrap model with DistributedDataParallel
model = DDP(model, device_ids=[local_rank])

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Delayed Gradient Averaging (DGA) function
def delayed_gradient_averaging(model, optimizer, delay=1):
    for param in model.parameters():
        if param.grad is not None:
            # Delay the receiving of averaged gradients by 'delay' steps
            if hasattr(param, 'grad_delayed'):
                # Apply correction to the current gradient
                param.grad.data = param.grad.data + param.grad_delayed - param.grad_received
            else:
                param.grad_delayed = torch.zeros_like(param.grad.data)
                param.grad_received = torch.zeros_like(param.grad.data)
            
            # Save the current gradients
            param.grad_delayed.copy_(param.grad.data)

            # Average the gradients across all processes
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

            # Store the received gradients
            param.grad_received.copy_(param.grad.data)

# Training loop with Delayed Gradient Averaging
def train(dataloader, model, optimizer, criterion, delay=1):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(local_rank), target.to(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Apply Delayed Gradient Averaging
        delayed_gradient_averaging(model, optimizer, delay=delay)

        # Step the optimizer
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Rank {local_rank}, Batch {batch_idx}, Loss: {loss.item()}")

# Dummy data loader
def get_dataloader(batch_size):
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 1024),
        torch.randint(0, 10, (1000,))
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=32)
    
    # Train with Delayed Gradient Averaging
    print("Training with Delayed Gradient Averaging")
    train(dataloader, model, optimizer, criterion, delay=1)
