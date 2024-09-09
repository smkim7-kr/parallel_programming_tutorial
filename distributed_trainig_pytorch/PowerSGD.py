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

# Function to perform PowerSGD compression
def power_sgd_compression(grad, rank=1):
    with torch.no_grad():
        # Reshape gradient to 2D matrix (if needed)
        original_shape = grad.shape
        grad_2d = grad.view(original_shape[0], -1)

        # Perform PowerSGD low-rank approximation
        # Step 1: Randomly initialize Q
        Q = torch.randn(grad_2d.shape[1], rank, device=grad.device)

        # Step 2: Multiply gradient by Q
        P = torch.mm(grad_2d, Q)

        # Step 3: Orthogonalize P (optional)
        P = torch.qr(P)[0]

        # Step 4: Recompute Q using the new P
        Q = torch.mm(grad_2d.t(), P)

        # Step 5: Low-rank gradient approximation
        approx_grad = torch.mm(P, Q.t())

        # Reshape back to original gradient shape
        approx_grad = approx_grad.view(original_shape)

        # Return the approximated gradient
        return approx_grad

# Training loop with PowerSGD compression
def train(dataloader, model, optimizer, criterion, rank=1):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(local_rank), target.to(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Apply PowerSGD compression to each parameter's gradient
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = power_sgd_compression(param.grad.data, rank=rank)

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
    train(dataloader, model, optimizer, criterion, rank=1)
