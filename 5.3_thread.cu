#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/*定义 const 指针(由于指针本身的值不能改变所以必须得初始化）*/
__global__ void what_is_my_id(unsigned int * const block,
		unsigned int * const thread,
		unsigned int * const warp,
		unsigned int * const calc_thread)
{
	/* Thread is is block index * block size + thread offset into the block */
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	block[thread_idx] = blockIdx.x;
	thread[thread_idx]= threadIdx.x;

	/* Calculate warp using built in variable warpSize */
	warp[thread_idx] = threadIdx.x / warpSize;

	calc_thread[thread_idx] = thread_idx;

}

/* Declare statically four arrays of ARRAY_SIZE each */

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main(void)
{
	/* Total thread count = 2 * 64 = 128 */
	const unsigned int num_blocks = 2;
	const unsigned int num_threads = 64;

	// Declare pointers fro GPU based params
	unsigned int * gpu_block;
	unsigned int * gpu_thread;
	unsigned int * gpu_warp;
	unsigned int * gpu_cal_thread;

	// Declare loop counter for use later
	unsigned int i;

	// Allocate four arrays on the GPU
	cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_warp, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_cal_thread, ARRAY_SIZE_IN_BYTES);


	// Execute the kernel
	what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp, gpu_cal_thread);

	// copy back the gpu results to the cpu
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_cal_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	// free the arrays on the GPU 
	cudaFree(gpu_block);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_cal_thread);

	// print 
	for (i=0; i<ARRAY_SIZE; i++)
	{
		printf("Calculated Thread: %d - Block: %d - Warp %d - Thread %d \n",
				cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
	}

}
