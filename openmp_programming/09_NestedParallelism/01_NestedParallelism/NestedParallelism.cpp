#include <stdio.h>
#include <omp.h>

int main() {
    // Enable nested parallelism
    omp_set_nested(1);

    printf("Nested Parallelism: %d\n", omp_get_nested());

    // Outer parallel region
    #pragma omp parallel num_threads(2)
    {
        int outer_thread_id = omp_get_thread_num();
        int outer_num_threads = omp_get_num_threads();

        printf("Outer region: Thread %d out of %d\n", outer_thread_id, outer_num_threads);

        // Nested parallel region inside the outer region
        #pragma omp parallel num_threads(3)
        {
            int inner_thread_id = omp_get_thread_num();
            int inner_num_threads = omp_get_num_threads();

            printf("  Inner region: Outer thread %d, Inner thread %d out of %d\n",
                   outer_thread_id, inner_thread_id, inner_num_threads);
        }
    }

    return 0;
}
