#include <iostream>
#include <vector>  
#include <omp.h> 

#define VECTOR_SIZE 1024*1024
#define NUM_THREADS 4

int main(void){
    std::vector<int> a(VECTOR_SIZE, 1);
    std::vector<int> b(VECTOR_SIZE, 2);
    std::vector<int> c(VECTOR_SIZE, 0);

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tID = omp_get_thread_num();
        for (int i = tID ; i < VECTOR_SIZE; i+= NUM_THREADS)
            c[i] = a[i] + b[i];
    }
    
    std::cout << c[100] << std::endl;
}
