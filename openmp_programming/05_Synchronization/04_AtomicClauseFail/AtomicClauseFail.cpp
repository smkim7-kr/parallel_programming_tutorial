#include <stdio.h>
#include <omp.h>

// Function to simulate a complex operation
int complex_operation(int x) {
    return x * x + 2 * x;
}

int main() {
    int sum = 0;

    // Parallel region where atomic will fail due to complex operation
    printf("===== Attempting to use atomic with complex operation =====\n");
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        // Trying to use atomic on a complex operation
        #pragma omp atomic
        sum += complex_operation(i);  // Atomic will not protect this correctly
        printf("Thread %d: i = %d, sum = %d (with atomic on complex operation)\n", omp_get_thread_num(), i, sum);
    }

    printf("Final sum (with atomic on complex operation): %d\n", sum);

    return 0;
}
