
# Introduction to OpenMP - Lecture Summary

### Instructor: Duksu Kim, HPC Lab., KOREATECH

## 1. Introduction to OpenMP
- **OpenMP** stands for Open Multi-Processing.
- An API for multi-platform shared memory multiprocessing programming in C, C++, and Fortran.
- Designed for scalable parallelism with high-level programming constructs like threads.

## 2. OpenMP vs Pthreads
- **Pthreads (POSIX Threads):**
  - Low-level API for thread-based parallelism.
  - Offers more flexibility but is complex to use.
- **OpenMP:**
  - High-level API that simplifies the implementation of parallelism.
  - Uses directives (pragma) to manage parallelism.

## 3. OpenMP Programming Model
- Follows a fork-join model:
  - The master thread forks a team of threads.
  - Threads run in parallel and then join back at an implicit barrier.
- Key concepts:
  - Parallel construction.
  - Sharing work among threads.
  - Declaring scope of variables.
  - Synchronization.

## 4. Hello World in OpenMP
Example code to demonstrate basic OpenMP usage:
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void main (int argc, char* argv[]){
    int thread_count = strtol(argv[1], NULL, 10);

    #pragma omp parallel num_threads(thread_count)
    {
        printf("[Thread %d] Hello OpenMP!\n", omp_get_thread_num());
    }
    return 0;
}
```
- When run with multiple threads, the code prints a "Hello OpenMP" message from each thread.

## 5. Using OpenMP
- **Pragma Directive:**
  - `#pragma omp parallel [clause list]` to specify code blocks for parallel execution.
- **Compiling OpenMP Code:**
  - Example command using `gcc`:
    ```
    gcc −g −Wall −fopenmp −o omp_hello omp_hello.c
    ```
