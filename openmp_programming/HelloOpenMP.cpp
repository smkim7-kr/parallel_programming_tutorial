#include <iostream>  // Use C++ standard library for I/O
#include <omp.h>     // OpenMP header


int main() {
    #pragma omp parallel 
    {
        // Use std::cout for thread-safe output in C++
		int tid = omp_get_thread_num();
        std::cout << "[Thread " << tid << "] Hello OpenMP!" << std::endl;
    }
    return 0;
}
