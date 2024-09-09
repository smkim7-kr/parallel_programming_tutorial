#include <iostream>
#include <omp.h>

int computeValue(int i) { return i; }

int main()
{
    int a[10] = { 0 };
    int b[10] = { 0 };

    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for (int i = 0; i < 10; i++)
            a[i] = computeValue(i);

        // implicit barrier here

        #pragma omp for
        for (int i = 0; i < 9; i++)
            b[i] = 2 * a[i + 1];
    }

    std::cout << "a = ";
    for (int i = 0; i < 10; i++)
        std::cout << a[i] << " ";
    std::cout << std::endl;

    std::cout << "b = ";
    for (int i = 0; i < 10; i++)
        std::cout << b[i] << " ";
    std::cout << std::endl;

    return 0;
}
