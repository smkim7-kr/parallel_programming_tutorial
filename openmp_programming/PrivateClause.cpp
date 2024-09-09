#include <stdio.h>
#include <omp.h>

int main() {
    int n = 4;
    int sum = 0;  // This variable will be private for each thread

    #pragma omp parallel for private(sum) num_threads(4)
    for (int i = 0; i < n; i++) {
        sum = i * i;
        printf("Thread %d calculated sum = %d for i = %d\n", omp_get_thread_num(), sum, i);
    }

    // The sum will not reflect changes made inside the parallel region
    printf("Sum after loop (private): %d\n", sum);
    return 0;
}
