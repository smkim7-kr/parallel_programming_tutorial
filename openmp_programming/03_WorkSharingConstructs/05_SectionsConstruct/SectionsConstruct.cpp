#include <stdio.h>
#include <omp.h>

int main() {
    // Case 1: Number of threads < Number of sections
    printf("Case 1: Number of threads < Number of sections\n");

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            // each section can execute different work
            #pragma omp section
            {
                printf("Thread %d is executing section 1\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                printf("Thread %d is executing section 2\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                printf("Thread %d is executing section 3\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                printf("Thread %d is executing section 4\n", omp_get_thread_num());
            }
        }
    }

    // Case 2: Number of threads > Number of sections
    printf("\nCase 2: Number of threads > Number of sections\n");

    #pragma omp parallel num_threads(4)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("Thread %d is executing section 1\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                printf("Thread %d is executing section 2\n", omp_get_thread_num());
            }
        }
    }

    return 0;
}
