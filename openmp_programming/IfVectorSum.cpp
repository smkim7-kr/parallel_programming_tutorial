#include <iostream>
#include <omp.h>
#include <mutex>

std::mutex printMutex;

void run(bool _parallel, int _numThreads)
{
    #pragma omp parallel num_threads(_numThreads) if(_parallel)
    {
        // Lock the mutex to ensure exclusive access to std::cout
        std::lock_guard<std::mutex> guard(printMutex);

        // Print the thread information
        std::cout << "I'm thread " << omp_get_thread_num()
                  << " out of " << omp_get_num_threads() << " thread(s)" << std::endl;
    }
}

int main()
{
    std::cout << "-- run(true, 4) --" << std::endl;
    run(true, 4);
    std::cout << "-- run(false, 4) --" << std::endl;
    run(false, 4);
}
