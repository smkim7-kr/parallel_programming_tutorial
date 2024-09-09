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

# 1-bit SGD quantization
def one_bit_sgd_quantization(grad):
    sign = torch.sign(grad)
    scale = torch.mean(torch.abs(grad))
    quantized_grad = sign * scale
    return quantized_grad

# Threshold quantization
def threshold_quantization(grad, threshold=0.1):
    quantized_grad = torch.zeros_like(grad)
    mask = torch.abs(grad) > threshold
    quantized_grad[mask] = torch.sign(grad[mask]) * threshold
    return quantized_grad

# TernGrad quantization
def terngrad_quantization(grad):
    max_val = torch.max(torch.abs(grad))
    tern_grad = torch.sign(grad) * max_val
    prob = torch.abs(grad) / max_val
    quantized_grad = torch.where(torch.rand_like(grad) < prob, tern_grad, torch.zeros_like(grad))
    return quantized_grad

# Training loop with gradient quantization techniques
def train(dataloader, model, optimizer, criterion, quantization_method):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(local_rank), target.to(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Apply selected quantization method to each parameter's gradient
        for param in model.parameters():
            if param.grad is not None:
                if quantization_method == "1-bit":
                    param.grad.data = one_bit_sgd_quantization(param.grad.data)
                elif quantization_method == "threshold":
                    param.grad.data = threshold_quantization(param.grad.data)
                elif quantization_method == "terngrad":
                    param.grad.data = terngrad_quantization(param.grad.data)

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
    
    # Train with 1-bit SGD quantization
    print("Training with 1-bit SGD Quantization")
    train(dataloader, model, optimizer, criterion, quantization_method="1-bit")
    
    # Train with Threshold Quantization
    print("Training with Threshold Quantization")
    train(dataloader, model, optimizer, criterion, quantization_method="threshold")
    
    # Train with TernGrad Quantization
    print("Training with TernGrad Quantization")
    train(dataloader, model, optimizer, criterion, quantization_method="terngrad")
