#include <stdio.h>
#include <omp.h>

int main() {
    const int n = 4;
    int shared_value = 0;  // This will be shared
    int private_value = 0; // This will be private for each thread

    #pragma omp parallel for default(none) shared(shared_value) private(private_value) num_threads(n)
    for (int i = 0; i < n; i++) {
        private_value = i + 1;
        #pragma omp critical
        {
            shared_value += private_value;
            printf("Thread %d added private_value %d to shared_value, shared_value now = %d\n",
                   omp_get_thread_num(), private_value, shared_value);
        }
    }

    printf("Final shared_value (default): %d\n", shared_value);
    return 0;
}
