#include <stdio.h>
#include <omp.h>

int main() {
    int n = 4;
    int sum = 0;  // This variable will be shared among threads

    #pragma omp parallel for shared(sum) num_threads(4)
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        {
            sum += i;
            printf("Thread %d added %d, sum now = %d\n", omp_get_thread_num(), i, sum);
        }
    }

    printf("Final sum (shared): %d\n", sum);
    return 0;
}
