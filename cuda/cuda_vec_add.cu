#include "device_launch_parameters.h"

#include <stdio.h>

#define SIZE 1024

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	int i = threadIdx.x;

	if (i < n)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	a = (int *)malloc(SIZE * sizeof(int));
	b = (int *)malloc(SIZE * sizeof(int));
	c = (int *)malloc(SIZE * sizeof(int));

	cudaMalloc(&d_a, SIZE * sizeof(int));
	cudaMalloc(&d_b, SIZE * sizeof(int));
	cudaMalloc(&d_c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// 1 block of threads, SIZE of total threads in the block
	VectorAdd <<< 1, SIZE >>> (d_a, d_b, d_c, SIZE);

	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < 10; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	free(a);
	free(b);
	free(c);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
