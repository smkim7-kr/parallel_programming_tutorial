#include <stdio.h>
#include <omp.h>

int main() {
    int i, N = 12;
    int array[N];

    // Initialize the array
    for (i = 0; i < N; i++) {
        array[i] = i + 1;
    }

    printf("Dynamic scheduling with chunk size 4:\n");
    #pragma omp parallel for schedule(dynamic, 4) 
    for (i = 0; i < N; i++) {
        printf("Thread %d: processing element %d\n", omp_get_thread_num(), array[i]);
    }

    printf("\nDynamic scheduling with chunk size 2:\n");
    #pragma omp parallel for schedule(dynamic, 2) 
    for (i = 0; i < N; i++) {
        printf("Thread %d: processing element %d\n", omp_get_thread_num(), array[i]);
    }

    return 0;
}
