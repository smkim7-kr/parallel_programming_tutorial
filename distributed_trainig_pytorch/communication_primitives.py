import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def broadcast_example(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
    dist.broadcast(tensor, src=0)
    print(f'Rank {rank} has data {tensor[0]} after Broadcast')

def gather_example(rank, size):
    tensor = torch.ones(1) * rank
    gather_list = [torch.zeros(1) for _ in range(size)] if rank == 0 else None
    dist.gather(tensor, gather_list, dst=0)
    if rank == 0:
        print(f'Rank {rank} has gathered data: {gather_list}')

def allreduce_example(rank, size):
    tensor = torch.ones(1) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'Rank {rank} has data {tensor[0]} after AllReduce')

def run(rank, size):
    setup(rank, size)

    print(f"Running broadcast_example on rank {rank}")
    broadcast_example(rank, size)

    print(f"\nRunning gather_example on rank {rank}")
    gather_example(rank, size)

    print(f"\nRunning allreduce_example on rank {rank}")
    allreduce_example(rank, size)

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run,
             args=(world_size,),
             nprocs=world_size,
             join=True)
