#include <iostream>
#include <omp.h>

void taskA() {
    std::cout << "Task A executed by thread " << omp_get_thread_num() << std::endl;
    // Task A code here
}

void taskB() {
    std::cout << "Task B executed by thread " << omp_get_thread_num() << std::endl;
    // Task B code here
}

void taskC() {
    std::cout << "Task C executed by thread " << omp_get_thread_num() << std::endl;
    // Task C code here
}

void taskD() {
    std::cout << "Task D executed by thread " << omp_get_thread_num() << std::endl;
    // Task D code here
}

void taskE() {
    std::cout << "Task E executed by thread " << omp_get_thread_num() << std::endl;
    // Task E code here
}

int main() {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            taskA();  // Task A is executed by one thread
        }

        #pragma omp section
        {
            taskB();  // Task B is executed by another thread
            taskE();
        }

        #pragma omp section
        {
            taskC();  // Task C is executed by another thread
        }

        #pragma omp section
        {
            taskD();  // Task D is executed by another thread
        }
    }

    return 0;
}
