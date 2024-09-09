#include <stdio.h>
#include <omp.h>

int main() {
    int n = 4;
    int last_value = 0;

    #pragma omp parallel for lastprivate(last_value) num_threads(4)
    for (int i = 0; i < n; i++) {
        last_value = i * 2;
        printf("Thread %d processed i = %d, last_value = %d\n", omp_get_thread_num(), i, last_value);
    }

    // After the loop, last_value will be the value from the last iteration
    printf("Final last_value (lastprivate): %d\n", last_value);
    return 0;
}
