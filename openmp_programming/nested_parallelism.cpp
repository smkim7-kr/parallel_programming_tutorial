#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    // Enable nested parallelism
    omp_set_nested(1);

    // Check if nested parallelism is enabled
    if (omp_get_nested()) {
        printf("Nested parallelism is enabled.\n");
    } else {
        printf("Nested parallelism is not enabled.\n");
        return 1;
    }

    #pragma omp parallel num_threads(2)
    {
        int outer_thread_id = omp_get_thread_num();
        int outer_num_threads = omp_get_num_threads();
        printf("Outer level: Thread %d of %d\n", outer_thread_id, outer_num_threads);

        #pragma omp parallel num_threads(2)
        {
            int inner_thread_id = omp_get_thread_num();
            int inner_num_threads = omp_get_num_threads();
            printf("  Inner level: Thread %d of %d (Outer thread %d)\n", inner_thread_id, inner_num_threads, outer_thread_id);
        }
    }

    return 0;
}
