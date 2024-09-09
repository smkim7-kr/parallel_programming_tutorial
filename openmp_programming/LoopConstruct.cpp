#include <iostream>
#include <omp.h>
#include <sstream>

int main()
{
    #pragma omp parallel num_threads(4)
    {
        // Create a thread-local buffer
        std::ostringstream oss;

        #pragma omp for
        for (int i = 0; i < 8; i++) {
            oss << "[Thread " << omp_get_thread_num() 
                << "] executes loop iteration " << i << std::endl;
        }

        // Print the buffered output once the loop is done
        std::cout << oss.str();
    }

    return 0;
}
