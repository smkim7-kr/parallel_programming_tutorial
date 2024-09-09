import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

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
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Function to apply gradient pruning (top-k pruning)
def gradient_pruning(model, k):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                # Flatten the gradients
                flat_grad = grad_data.view(-1)
                # Find the threshold for top-k gradients
                threshold = torch.topk(torch.abs(flat_grad), k)[0][-1]
                # Zero out gradients smaller than the threshold
                grad_data[torch.abs(grad_data) < threshold] = 0

# Training loop with sparse communication using gradient pruning
def train(dataloader, model, optimizer, criterion, k=1000):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(local_rank), target.to(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Apply gradient pruning
        gradient_pruning(model, k=k)

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
    train(dataloader, model, optimizer, criterion, k=1000)
