#include <iostream>
#include <omp.h>

int main()
{
    const int n = 4;
    const int size = 5; // Size of each array
    int arr1[size] = {0};
    int arr2[size] = {0};
    int arr3[size] = {0};
    int arr4[size] = {0};

    #pragma omp parallel num_threads(n)
    {
        #pragma omp for
        for (int tid = 0; tid < n; tid++) {
            switch (tid) {
                case 0:
                    for (int i = 0; i < size; i++) {
                        arr1[i] = i + 1; // Initialize arr1
                    }
                    std::cout << "Thread " << omp_get_thread_num() << " initialized arr1" << std::endl;
                    break;

                case 1:
                    for (int i = 0; i < size; i++) {
                        arr2[i] = (i + 1) * 2; // Initialize arr2
                    }
                    std::cout << "Thread " << omp_get_thread_num() << " initialized arr2" << std::endl;
                    break;

                case 2:
                    for (int i = 0; i < size; i++) {
                        arr3[i] = (i + 1) * 3; // Initialize arr3
                    }
                    std::cout << "Thread " << omp_get_thread_num() << " initialized arr3" << std::endl;
                    break;

                case 3:
                    for (int i = 0; i < size; i++) {
                        arr4[i] = (i + 1) * 4; // Initialize arr4
                    }
                    std::cout << "Thread " << omp_get_thread_num() << " initialized arr4" << std::endl;
                    break;

                default:
                    std::cout << "Thread " << omp_get_thread_num() << " is performing an unknown task" << std::endl;
                    break;
            }
        }
    }

    // Print the initialized arrays
    std::cout << "arr1: ";
    for (int i = 0; i < size; i++) std::cout << arr1[i] << " ";
    std::cout << std::endl;

    std::cout << "arr2: ";
    for (int i = 0; i < size; i++) std::cout << arr2[i] << " ";
    std::cout << std::endl;

    std::cout << "arr3: ";
    for (int i = 0; i < size; i++) std::cout << arr3[i] << " ";
    std::cout << std::endl;

    std::cout << "arr4: ";
    for (int i = 0; i < size; i++) std::cout << arr4[i] << " ";
    std::cout << std::endl;

    return 0;
}
