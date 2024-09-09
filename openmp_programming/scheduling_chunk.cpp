#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 100000000  // Size of the array

// Function to generate an array with random numbers
void generate_random_array(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;  // Random numbers between 0 and 99
    }
}

int main() {
    int *array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    generate_random_array(array, ARRAY_SIZE);

    double start_time, end_time;
    long long sum;
    int chunk_sizes[] = {1, 10, 100, 1000, 10000, 100000};  // Different chunk sizes

    for (int i = 0; i < sizeof(chunk_sizes)/sizeof(chunk_sizes[0]); i++) {
        int chunk_size = chunk_sizes[i];

        sum = 0;
        start_time = omp_get_wtime();
        #pragma omp parallel for reduction(+:sum) schedule(static, chunk_size)
        for (int j = 0; j < ARRAY_SIZE; j++) {
            sum += array[j];
        }
        end_time = omp_get_wtime();
        printf("Chunk size: %d\n", chunk_size);
        printf("Sum: %lld\n", sum);
        printf("Time taken with chunk size %d: %f seconds\n", chunk_size, end_time - start_time);
        printf("--------------------------------------------\n");
    }

    free(array);
    return 0;
}
