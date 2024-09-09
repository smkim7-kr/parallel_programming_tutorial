
# Introduction to Parallel Computing - Lecture Summary

### Instructor: Duksu Kim, HPC Lab., KOREATECH

## 1. Parallel Architecture
- Overview of parallel architectures:
  - Intel Quad-core i7
  - Sony PlayStation
  - IBM Blue Gene
  - Nvidia Kepler architecture
  - Google data centers
- Comparison between parallel computing and distributed computing.

## 2. Flynn’s Taxonomy
- **SISD (Single Instruction, Single Data):** Traditional single-core processors.
- **SIMD (Single Instruction, Multiple Data):** 
  - Vector processors, GPUs.
  - Technologies: MXX/SSE/AVX(x86), XeonPhi.
- **MISD (Multiple Instruction, Single Data):** Not commonly covered.
- **MIMD (Multiple Instruction, Multiple Data):** 
  - Multi-core CPUs.
  - Represents thread-level parallelism.

## 3. SIMD Parallelism
- **SIMD**: Single Instruction, Multiple Data.
- Examples of vector processors.
- Challenges:
  - Memory access patterns.
  - Bank conflicts.
  - Divergence.

## 4. MIMD Parallelism
- **MIMD**: Multiple Instruction, Multiple Data.
- Challenges:
  - Nondeterminism.
  - Race conditions.
  - Need for synchronization (e.g., mutex locks).

## 5. Shared Memory vs. Distributed Memory Systems
- Differences between shared and distributed memory systems.
- Implications for parallel computing.

## 6. Performance of Parallel Computing
- **Speedup**: Concept of linear speedup.
- **Efficiency**: Consideration of overheads.
- **Amdahl’s Law**: Limits of speedup due to the sequential portion of tasks.

## 7. Scalability
- Importance of scalability in parallel systems.
- Focus on high scalability for improved performance.
