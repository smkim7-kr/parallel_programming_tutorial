#include <stdio.h>
#include <omp.h>

int main() {
    const int size = 5;
    int array1[size] = {0};  // Array to be initialized by one thread
    int array2[size] = {1, 2, 3, 4, 5};  // Array used by all threads

    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();

        // Single thread initializes array1, others continue without waiting
        #pragma omp single nowait
        {
            printf("Thread %d is initializing array1\n", thread_id);
            for (int i = 0; i < size; i++) {
                array1[i] = i * 10;
            }
            printf("Thread %d completed initializing array1\n", thread_id);
        }

        // All threads perform computations on array2 immediately
        printf("Thread %d is performing computations on array2\n", thread_id);
        for (int i = 0; i < size; i++) {
            array2[i] *= 2;
            printf("Thread %d: array2[%d] = %d\n", thread_id, i, array2[i]);
        }
    }

    // Print the final state of array1 and array2
    printf("\nFinal array1: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array1[i]);
    }
    printf("\n");

    printf("Final array2: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array2[i]);
    }
    printf("\n");

    return 0;
}
