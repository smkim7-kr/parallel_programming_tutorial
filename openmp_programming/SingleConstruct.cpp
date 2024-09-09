#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        
        // Code executed by all threads
        std::cout << "Thread " << thread_id << " is doing work before the single block" << std::endl;

        #pragma omp single
        {
            // This block is executed by only one thread
            std::cout << "Thread " << thread_id << " is executing the single block" << std::endl;
            // Example task
            std::cout << "Only one thread is performing this task." << std::endl;
        }

        // Code executed by all threads after the single block
        std::cout << "Thread " << thread_id << " is doing work after the single block" << std::endl;
    }

    return 0;
}
