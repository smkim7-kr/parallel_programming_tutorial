
# Lecture 18 Summary

## Course: MIT 6.5940: TinyML and Efficient Deep Learning Computing

### Lecture Title: Distributed Training

### Instructor: Song Han, Associate Professor, MIT

---

## Lecture Plan

1. **Hybrid (Mixed) Parallelism and Auto-Parallelization**
   - Explore hybrid parallelism approaches.
   - Methods to automatically parallelize deep learning models.

2. **Bandwidth and Latency Bottlenecks in Distributed Training**
   - Identify and understand bottlenecks in bandwidth and latency during distributed training.

3. **Gradient Compression: Overcoming Bandwidth Bottlenecks**
   - Techniques such as Gradient Pruning and Quantization.
   - Implementations like Sparse Communication, Deep Gradient Compression, 1-Bit SGD, and TernGrad.

4. **Delayed Gradient Update: Overcoming Latency Bottlenecks**
   - Approaches to mitigate latency issues in distributed training.

---

## Comparing Different Parallelization Strategies

- **Data Parallelism**
  - Splits data across multiple GPUs.
  - Involves multiple copies of the model.
  - High utilization, but also high memory cost and low communication.

- **Tensor Parallelism**
  - Model is split by tensor.
  - Involves a single copy of the model.
  - Balances utilization with lower memory cost but higher communication needs.

- **Pipeline Parallelism**
  - Model is divided by layers.
  - Involves a single copy of the model.
  - Low utilization and memory cost with medium communication requirements.

---

## Hybrid Parallelism

- **2D Parallelism**
  - Combines Data Parallelism and Pipeline Parallelism.
  - Groups GPUs for different parallelism strategies.

- **3D Parallelism**
  - Incorporates Pipeline, Tensor, and Data Parallelism.
  - Designed for extreme-scale model training.

---

## Auto Parallelization Techniques

- **Motivation**
  - Addressing models too large for a single GPU by splitting them across multiple GPUs.
  - Splitting layers for efficient processing.

- **Strategies**
  - **Inter-operator Parallelism:** Splitting computation across different devices.
  - **Intra-operator Parallelism:** Further breaking down operations within devices.

---

## Alpa: Automating Parallelism

- **Unified Compiler**
  - Alpa automates both inter- and intra-operator parallelism.
  - Optimizes the search space for the best parallel strategies.

- **Evaluation**
  - Alpa matches and even surpasses manually optimized systems in performance.

---

## Bottlenecks in Distributed Training

- **Communication Challenges**
  - Synchronization requirements and high communication frequency can create bottlenecks.

- **Model Size and Transfer Data**
  - Larger models require more data transfer, leading to longer transfer times.

- **Network Bandwidth**
  - Insufficient bandwidth can severely limit training efficiency.

- **Network Latency**
  - High latency, especially over long distances or wireless connections, slows down training.

---

## Gradient Compression Techniques

- **Gradient Pruning**
  - Sends only the most significant gradients.
  - Keeps unpruned gradients as residuals for future updates.

- **Momentum and Gradient Compression**
  - Momentum-based optimizers need correction when using gradient pruning.

- **Warm-Up Techniques**
  - Gradually increasing learning rate and sparsity helps the optimizer adapt.

- **PowerSGD**
  - Low-rank gradient compression method that maintains high accuracy while reducing communication costs.

---

## Bandwidth vs. Latency

- **Improving Bandwidth**
  - Techniques like gradient compression, quantization, and hardware upgrades can enhance bandwidth.

- **Challenges with Latency**
  - Physical limits and network congestion make latency difficult to reduce.

---

## Delayed Gradient Averaging (DGA)

- **Key Idea**
  - Allows stale gradients to overlap communication with computation, improving training throughput.

- **Effectiveness**
  - DGA shows significant speedup in real-world benchmarks without degrading model performance.

---

## Summary

- **Hybrid Parallelism and Auto-Parallelization**
- **Understanding Bandwidth and Latency Bottlenecks**
- **Gradient Compression Strategies**
- **Overcoming Latency with Delayed Gradient Updates**

---

## References

- Various research papers and methods discussed throughout the lecture are referenced.

