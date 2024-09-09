
# MIT 6.5940: TinyML and Efficient Deep Learning Computing

## Lecture 17: Distributed Training

**Instructor:** Song Han, Associate Professor, MIT  
**Affiliations:** MIT & NVIDIA  

---

## Lecture Plan

1. Background and motivation
2. Parallelization methods for distributed training
3. Data parallelism
4. Communication primitives
5. Reducing memory in data parallelism: ZeRO-1 / 2 / 3 and FSDP
6. Pipeline parallelism
7. Tensor parallelism

---

## Background and Motivation

As machine learning models, particularly in computer vision and natural language processing (NLP), become increasingly complex, the computational resources required to train these models grow exponentially. Distributed training enables the parallelization of model training across multiple devices, reducing the time and resource bottleneck.

### The Growing Complexity of Models

- **Vision Models:** Historically, improvements in model performance on tasks like ImageNet have come with increased computational demands.
- **NLP Models:** NLP models, such as GPT and BERT, have seen an exponential increase in the number of parameters, with GPT-3 reaching 175 billion parameters. Training such models on a single GPU would be impractically slow, taking hundreds of years.

### Necessity of Distributed Training

- **Developer Efficiency:** Reducing the time to train models from days to minutes with distributed training drastically accelerates research and development cycles.
- **Example:** Training a large-scale video model on the SUMMIT supercomputer can reduce the training time from days to minutes.

---

## Parallelization Methods

### Data Parallelism

Data parallelism involves splitting the dataset across multiple GPUs, each with its own replica of the model. The GPUs process different portions of the data in parallel, and gradients are aggregated to update the model weights.

**Steps in Data Parallelism:**
1. **Replicate Models to Each Worker:** Each GPU has a local replica of the neural network model.
2. **Split Training Data:** The training data is split among GPUs.
3. **Compute Gradients:** Each GPU computes gradients locally.
4. **Aggregate and Synchronize Gradients:** Gradients are aggregated and synchronized across all GPUs.
5. **Update Model Parameters:** The model parameters are updated with the aggregated gradients.

### Pipeline Parallelism

Pipeline parallelism splits the model itself across different GPUs, with each GPU processing a different stage of the model. This method is particularly useful for very large models that cannot fit into the memory of a single GPU.

- **Naive Pipeline Parallelism:** Suffered from low GPU utilization due to the sequential dependency between different layers of the model.
- **Improved Pipeline Parallelism (GPipe):** Introduced micro-batches to increase utilization, improving from 25% to 57%.

### Tensor Parallelism

Tensor parallelism divides the tensors (e.g., weight matrices) within the model across multiple GPUs. This method allows for fine-grained parallelization, which can significantly increase GPU utilization in large-scale models.

- **First Linear Layer Partitioning:** The input is broadcasted to all GPUs, and the computation is split among them.
- **Second Linear Layer Partitioning:** The results are gathered using AllReduce to combine outputs from all GPUs.

---

## Communication Primitives

Efficient communication between GPUs is crucial in distributed training. The following primitives are commonly used:

- **One-to-One Communication:** Send and Receive operations between individual GPUs.
- **One-to-Many Communication:**
  - **Scatter:** Distributes data from one GPU to multiple GPUs.
  - **Gather:** Collects data from multiple GPUs to one GPU.
  - **Reduce:** Aggregates data (e.g., summing gradients) from multiple GPUs.
  - **Broadcast:** Sends identical data to all GPUs.
- **Many-to-Many Communication:** 
  - **All-Reduce:** Combines data from all GPUs and distributes the result back to all GPUs.
  - **All-Gather:** Gathers data from all GPUs and distributes the combined data back to all GPUs.

### Optimization Techniques

- **Ring-AllReduce:** A communication strategy that reduces bandwidth usage and time complexity compared to sequential methods.
- **Recursive Halving AllReduce:** Further optimizes AllReduce by reducing the number of communication steps to log(N), where N is the number of GPUs.

---

## Memory Optimization in Data Parallelism: ZeRO and FSDP

To handle extremely large models, such as those with hundreds of billions of parameters, memory optimization is essential.

### ZeRO: Zero Redundancy Optimizer

- **ZeRO-1:** Partitions optimizer states across GPUs, reducing memory usage.
- **ZeRO-2:** Extends ZeRO-1 by also partitioning gradients.
- **ZeRO-3:** Further reduces memory usage by partitioning both weights and gradients, allowing the training of models with up to 320 billion parameters using 64 GPUs.

### Fully Sharded Data Parallel (FSDP)

In PyTorch, ZeRO-3 is implemented via FSDP, enabling highly efficient training of very large models by fully sharding all model states (weights, gradients, and optimizer states) across GPUs.

---

## Summary of Different Parallelisms

- **Data Parallelism:** Best for models that can be easily replicated across GPUs but may have communication bottlenecks.
- **Pipeline Parallelism:** Suitable for very large models that require splitting across multiple GPUs but may suffer from under-utilization.
- **Tensor Parallelism:** Offers fine-grained parallelism by splitting model tensors across GPUs, providing high utilization with some communication overhead.

---

## References

- TSM: Temporal Shift Module for Efficient Video Understanding [Lin et al. 2019]
- Scaling Distributed Machine Learning with the Parameter Server [Li et al. 2014]
- Optimization of Collective Communication Operations [Rajeev et al. 2005]
- GPipe: Efficient Training of Giant Neural Networks [Huang et al. 2018]
- Megatron-LM: Training Multi-Billion Parameter Language Models [Shoeybi et al. 2019]
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models [Rajbhandari et al. 2019]
