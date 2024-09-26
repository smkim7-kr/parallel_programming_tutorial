#include <stdio.h>
#include <omp.h>

int main() {
    int i, N = 100;
    int array[N];
    int sum_without_reduction = 0;
    int sum_with_reduction = 0;

    // Initialize the array with values 1 to N
    for (i = 0; i < N; i++) {
        array[i] = i + 1;
    }

    // ==========================
    // Parallel region without reduction (prone to race conditions)
    printf("===== Without Reduction Clause =====\n");
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        sum_without_reduction += array[i];  // Race conditions may occur here
    }
    printf("Total sum (without reduction): %d\n", sum_without_reduction);

    // ==========================
    // Parallel region with reduction to avoid race conditions
    printf("\n===== With Reduction Clause =====\n");
    #pragma omp parallel for reduction(+:sum_with_reduction)
    // each thread gets its private sum_with_rediction variable
    for (i = 0; i < N; i++) {
        sum_with_reduction += array[i];  // Safe reduction across threads
    }
    // safe total sum into the shared variable
    printf("Total sum (with reduction): %d\n", sum_with_reduction);

    return 0;
}
