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

    // Static scheduling
    sum = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) schedule(static, 1)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += array[i];
    }
    end_time = omp_get_wtime();
    printf("Sum with static scheduling: %lld\n", sum);
    printf("Time taken with static scheduling: %f seconds\n", end_time - start_time);
    printf("--------------------------------------------\n");

    // Dynamic scheduling
    sum = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) schedule(dynamic, 1)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += array[i];
    }
    end_time = omp_get_wtime();
    printf("Sum with dynamic scheduling: %lld\n", sum);
    printf("Time taken with dynamic scheduling: %f seconds\n", end_time - start_time);
    printf("--------------------------------------------\n");

    // Guided scheduling
    sum = 0;
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) schedule(guided, 1)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sum += array[i];
    }
    end_time = omp_get_wtime();
    printf("Sum with guided scheduling: %lld\n", sum);
    printf("Time taken with guided scheduling: %f seconds\n", end_time - start_time);
    printf("--------------------------------------------\n");

    free(array);
    return 0;
}
