#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>


int main(void) {
	int tID = 0;
	#pragma omp parallel private (tID)
	{
		tID = omp_get_thread_num();

		if (tID % 2 == 0) sleep(5);
		printf("[%d] before\n", tID);

		#pragma omp barrier

		printf("[%d] after\n", tID);
	}
	
	return 0;
}