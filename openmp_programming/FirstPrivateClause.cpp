#include <stdio.h>
#include <omp.h>

int main() {
    int n = 4;
    int initial_value = 10;

    #pragma omp parallel for firstprivate(initial_value) num_threads(4)
    for (int i = 0; i < n; i++) {
        initial_value += i;
        printf("Thread %d started with initial_value = %d, updated to %d\n", omp_get_thread_num(), 10, initial_value);
    }

    // initial_value will still be 10 outside the loop
    printf("Initial value after loop (firstprivate): %d\n", initial_value);
    return 0;
}
