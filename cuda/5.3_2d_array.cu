#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE_X * ARRAY_SIZE_Y))

/*定义 const 指针(由于指针本身的值不能改变所以必须得初始化）*/
__global__ void what_is_my_id_2d_A(
	unsigned int * const block_x,
	unsigned int * const block_y,
	unsigned int * const thread,
	unsigned int * const calc_thread,
	unsigned int * const x_thread,
	unsigned int * const y_thread,
	unsigned int * const grid_dimx,
	unsigned int * const block_dimx,
	unsigned int * const grid_dimy,
	unsigned int * const block_dimy)
{	
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x; // 0~W-1
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y; // 0~H-1
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx; // global 

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread[thread_idx] = threadIdx.x;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;

}

/* Declare statically 6 arrays of ARRAY_SIZE each */
// the image is 32 x 16
unsigned int  cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_x_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_y_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int  cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(void)
{
	/* Total thread count of one block = 2 * 64 = 128 */
	// Total thread of all grid 128*4=512
	const dim3 threads_rect(32, 4); // W=32, H=4
	const dim3 blocks_rect(1, 4); // it means W=1, H=4


	// another shape
	const dim3 threads_square(16, 8);
	const dim3 blocks_square(2, 2);

	// Declare pointers fro GPU based params
	unsigned int *  gpu_block_x;
	unsigned int *  gpu_block_y;
	unsigned int *  gpu_thread;
	unsigned int *  gpu_calc_thread;
	unsigned int *  gpu_x_thread;
	unsigned int *  gpu_y_thread;
	unsigned int *  gpu_grid_dimx;
	unsigned int *  gpu_block_dimx;
	unsigned int *  gpu_grid_dimy;
	unsigned int *  gpu_block_dimy;

	// Allocate four arrays on the GPU
	cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_x_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_y_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);


	// Execute the kernel
	for(int kernel=0; kernel < 2; kernel++)
	{
		switch(kernel)
		{
			case 0:
			{
				what_is_my_id_2d_A<<<blocks_rect, threads_rect>>>(
					gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread,
					gpu_x_thread, gpu_y_thread, gpu_grid_dimx, gpu_block_dimx,
					gpu_grid_dimy, gpu_block_dimy
				);
			} break;

			case 1:
			{
				what_is_my_id_2d_A<<<blocks_square, threads_square>>>(
					gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread,
					gpu_x_thread, gpu_y_thread, gpu_grid_dimx, gpu_block_dimx,
					gpu_grid_dimy, gpu_block_dimy
				);
			} break;

			default: exit(1); break;

		}
	}


	// copy back the gpu results to the cpu
	cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_x_thread, gpu_x_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_y_thread, gpu_y_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	// free the arrays on the GPU 
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_x_thread);
	cudaFree(gpu_y_thread);
	cudaFree(gpu_grid_dimx);
	cudaFree(gpu_block_dimx);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);

	// print 
	for (int y=0; y <ARRAY_SIZE_Y; y++)
	{
		for (int x=0; x < ARRAY_SIZE_X; x++)
		{
			printf("CT: %d - Block_X-%d, Block_Y-%d, Thread_X-%d, Thread_Y-%d, Thread-%d,Grid_dimx-%d, Block_dimx-%d, Grid_dimy-%d, Block_dimy-%d\n",
					cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x], cpu_x_thread[y][x], cpu_y_thread[y][x],
			cpu_thread[y][x], cpu_grid_dimx[y][x], cpu_block_dimx[y][x], cpu_grid_dimy[y][x],cpu_block_dimy[y][x]);
		}
	}

}
