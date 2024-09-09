
# Detailed Summary of Lecture on "Why Parallel Computing?"

## Course Information
- **Course**: Multicore Programming (CPH 351)
- **Instructor**: Duksu Kim, HPC Lab., KOREATECH

## Introduction to Parallel Computing
- **Parallel Computing vs Concurrent Computing**:
  - **Parallel Computing**: Involves multiple processes running simultaneously, typically on multiple processors or cores. The focus is on performing many calculations simultaneously to speed up computation.
  - **Concurrent Computing**: Involves multiple processes making progress, but not necessarily simultaneously. The focus is on managing multiple tasks, potentially within the same process or on a single core.

- **Importance of Parallel Computing**:
  - The lecture emphasizes the necessity of parallel computing in modern systems due to limitations in CPU clock frequency increases. As clock speeds plateau, performance gains must come from parallelism.
  - **Application Perspective**: Programs must be designed or adapted to take advantage of multiple cores/processors.
  - **Architecture Perspective**: Hardware must support efficient parallel execution, including memory access patterns, communication between cores, and load balancing.

## Fundamental Concepts
- **Speedup**: 
  - **Formula**: 
    \[
    	ext{Speedup} = rac{	ext{Performance with } p 	ext{ processors}}{	ext{Performance with 1 processor}}
    \]
  - This metric helps to quantify the effectiveness of parallel execution. The lecture explains how this speedup is influenced by both the application and the architecture.

## Historical Context
- **Evolution of CPUs**: 
  - Example of transistor count:
    - **Intel 8080 (1978)**: 29K transistors.
    - **Intel Pentium 4 (1999)**: 28M transistors.
  - The exponential growth in hardware complexity necessitates more sophisticated software that can leverage multiple processing units effectively.

## Example Algorithms
1. **Serial Algorithm Example**:
   - A simple loop to calculate a sum sequentially:
   ```c
   int sum = 0;
   for (int i = 0; i < n; i++) {
       int x = ComputeNextValue();
       sum += x;
   }
   ```
   - **Explanation**: This code runs in a single thread, processing each value one at a time. The sum is updated with each computed value.

2. **Naïve Parallel Algorithm**:
   - Parallelized version where work is divided among multiple threads:
   ```c
   int ComputeMySum(int tid) {
       int my_sum = 0;
       for (int i = my_first; i < my_end; i++) {
           int my_x = ComputeNextValue();
           my_sum += my_x;
       }
       return my_sum;
   }

   // Main parallel execution
   if (tid == 1) {  // Master thread
       sum = my_sum[0];
       for (int i = 1; i < p; i++) {
           sum += receive(i);
       }
   } else {  // Slave threads
       sendMySum();
   }
   ```
   - **Explanation**: 
     - Each thread computes a partial sum (`my_sum`) for its assigned portion of the data. 
     - The master thread (tid = 1) collects these partial sums from all threads and computes the global sum.
     - This approach assumes each thread handles an equal portion of the data (`n/p` elements), but does not account for potential load imbalance or communication overhead.

   - **Example Execution**: 
     - Suppose there are 8 cores and `n = 24`. The data is split across cores:
       ```
       Core 0: [1,4,3], Core 1: [9,2,8], Core 2: [5,1,1], Core 3: [5,2,7],
       Core 4: [2,5,0], Core 5: [4,1,8], Core 6: [6,5,1], Core 7: [2,3,9]
       ```
     - Each core computes a partial sum, e.g., Core 0 calculates `8`, Core 1 calculates `19`, and so on.
     - The global sum is `95`.

3. **Improved Parallel Algorithm**:
   - Optimization of the naive approach to reduce computational time and overhead:
   - **Approach**: Introduces better load balancing, efficient communication, and possibly reducing the number of synchronization points between threads.
   - **Performance Analysis**:
     - The lecture compares the total computing time of the naive and improved algorithms.
     - A graph is provided showing how computing time decreases as the number of cores increases, but also showing diminishing returns.

## Performance Analysis
- **Comparison of Naïve and Improved Algorithms**:
  - **Naïve Algorithm**: Effective but has communication overhead and load imbalance.
  - **Improved Algorithm**: Achieves better performance by optimizing the distribution of tasks and reducing unnecessary communication.
  - **Graphical Analysis**: The lecture includes graphs that plot the number of cores against total computing time. The improved algorithm shows significant efficiency gains, particularly when moving from a small to a moderate number of cores.

## Discussion Points
- **Challenges in Parallel Computing**:
  - **Synchronization Overhead**: Managing data dependencies between threads introduces overhead.
  - **Load Imbalance**: If tasks are not evenly distributed, some cores may finish early while others continue working, leading to inefficiencies.
  - **Scalability**: While adding more cores generally reduces computing time, the benefits decrease as the number of cores increases beyond a certain point due to overhead and communication costs.

## Conclusion
- **Why Parallel Computing?**:
  - The lecture concludes by reinforcing the importance of parallel computing to overcome the limitations of traditional, serial computing models.
  - As hardware evolves with more cores and complex architectures, understanding and effectively applying parallel computing principles becomes crucial for high-performance applications.
