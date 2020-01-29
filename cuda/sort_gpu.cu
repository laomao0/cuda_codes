/*
 * 	 A codes of example of book [CUDA Programming, A developer's Guide to Parallel Computing with GPUs].
 * 	 Section 6.4 sorting a sequence using shared memory.
	 We sort a sequence independent into multiple sorted lists.
	 Then use a host function to combine those lists.
 */

#include <stdio.h>
#include "gpu_timer.h"


typedef unsigned int u32;

#define ARRAY_SIZE 6
#define ARRAY_SIZE_IN_BYTES (sizeof(u32) * (ARRAY_SIZE))
#define MAX_NUM_LISTS 2
#define NUM_LIST MAX_NUM_LISTS
#define NUM_ELEM ARRAY_SIZE

__device__ void radix_sort(u32 * const sort_tmp,
							const u32 num_lists,
							const u32 num_elements,
							u32 * const sort_tmp_0,
							u32 * const sort_tmp_1,
							const u32 tid)
{
//	u32 tid = threadIdx.x;

	printf("Thread ID: %d\n", tid);
	printf("Num of Elements %d\n", num_elements);
	printf("Num of Num_Lists %d\n", num_lists);


	// Sort into num_list, lists
	for (u32 bit=0; bit<32; bit++)  //32
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for (u32 i = 0; i < num_elements; i+= num_lists)
		{
			const u32 elem = sort_tmp[i+tid];
			const u32 bit_mask = ( 1 << bit);

			printf("ort_tmp[i+tid]: %d, i+tid: %d\n", elem, i+tid);

			if ((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1 + tid] = elem;
				base_cnt_1 += num_lists;

				printf("ort_tmp_1[base_cnt_1 + tid]: %d, base_cnt_1 + tid: %d\n", elem, base_cnt_1 + tid);
				printf("base_cnt_1: %d\n", base_cnt_1);

			}
			else
			{
				sort_tmp_0[base_cnt_0 + tid] = elem;
				base_cnt_0 += num_lists;

				printf("ort_tmp_0[base_cnt_0 + tid]: %d, base_cnt_0 + tid: %d\n", elem, base_cnt_0 + tid);
				printf("base_cnt_0: %d\n", base_cnt_0);
			}
		}

		for(u32 i=0; i<base_cnt_0; i+= num_lists)
		{
			sort_tmp[i+tid] = sort_tmp_0[i+tid];
		}

		for(u32 i=0; i<base_cnt_1; i+= num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}

	}

	__syncthreads();
}


/* generate num_lists independent sorted sequence */
// device function can be called from other device or global functions.
// device functions cannot be called from host code.
__device__ void radix_sort2(u32 * const sort_tmp,
							const u32 num_lists,
							const u32 num_elements,
							u32 * const sort_tmp_1,
							const u32 tid)
{
//	u32 tid = threadIdx.x;

	printf("Thread ID: %d\n", tid);
	printf("Num of Elements %d\n", num_elements);
	printf("Num of Num_Lists %d\n", num_lists);


	// Sort into num_list, lists
	for (u32 bit=0; bit<32; bit++)  //32
	{
		const u32 bit_mask = ( 1 << bit);
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for (u32 i = 0; i < num_elements; i+= num_lists)
		{
			const u32 elem = sort_tmp[i+tid];

			if ((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1 + tid] = elem;
				base_cnt_1 += num_lists;

			}
			else
			{
				sort_tmp[base_cnt_0 + tid] = elem;
				base_cnt_0 += num_lists;

			}
		}

		for(u32 i=0; i<base_cnt_1; i+= num_lists)
		{
			sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
		}

	}

	// wait for all threads done
	__syncthreads();
}


u32 find_min(const u32 * const src_array,
			u32 * const list_indexes,
			const u32 num_lists,
			const u32 num_elements_per_list)
{
	u32 min_val = 0xFFFFFFFF;
	u32 min_idx = 0;
	
	// Iterate over each of the lists
	for (u32 i=0; i<num_lists; i++)
	{
		// If the current list has already been emptied
		// then ignore it.
		if (list_indexes[i] < num_elements_per_list)
		{
			const u32 src_idx = i + (list_indexes[i] * num_lists);
			const u32 data = src_array[src_idx];

			if (data <= min_val)
			{
				min_val = data;
				min_idx = i;
			}
		}
	}

	list_indexes[min_idx]++;
	return min_val;
}

void merge_array(const u32 * const src_array,
				u32 * const dest_array,
				const u32 num_lists,
				const u32 num_elements)
{
	const u32 num_elements_per_list = (num_elements / num_lists);
	u32 list_indexes[MAX_NUM_LISTS];

	for (u32 list=0; list < num_lists; list++)
	{
		list_indexes[list] = 0;
	}

	for( u32 i=0; i < num_elements; i++)
	{
		dest_array[i] = find_min(src_array,
								list_indexes,
								num_lists,
								num_elements_per_list);
	}

}


__device__ void copy_data_to_shared(
		const u32 * const data,
		 u32 * const sort_tmp,
		const u32 num_lists,
		const u32 num_elements,
		const u32 tid)
{
	for(u32 i =0; i<num_elements; i+=num_lists)
	{
		sort_tmp[i+tid] = data[i+tid];
	}
	__syncthreads();
}

__global__ void gpu_sort_array(
	u32 * const data,
	const u32 num_lists,
	const u32 num_elements
)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	printf("Thread ID: %d\n", tid);

	__shared__ u32 sort_tmp[NUM_ELEM];
	__shared__ u32 sort_tmp_1[NUM_ELEM];

	copy_data_to_shared(data, sort_tmp, num_lists, num_elements, tid);

	radix_sort2(sort_tmp, num_lists, num_elements, sort_tmp_1, tid);

	//copy back
	copy_data_to_shared(sort_tmp, data, num_lists, num_elements, tid);

}



int main() {

	// CPU
	u32 DATA[ARRAY_SIZE] = {122, 10, 2, 22, 12, 9};
	u32 RESULT[ARRAY_SIZE];
//	u32 SORT_TMP_0[NUM_ELEM] = {0};
//	u32 SORT_TMP_1[NUM_ELEM] = {0};

	// GPU
	u32 * GPU_DATA;
//	u32 * GPU_SORT_TMP_0;
//	u32 * GPU_SORT_TMP_1;

	// Allocate four arrays on the GPU
	cudaMalloc((void **)&GPU_DATA, ARRAY_SIZE_IN_BYTES);
//	cudaMalloc((void **)&GPU_SORT_TMP_0, ARRAY_SIZE_IN_BYTES);
//	cudaMalloc((void **)&GPU_SORT_TMP_1, ARRAY_SIZE_IN_BYTES);

	cudaMemcpy(GPU_DATA, DATA, ARRAY_SIZE_IN_BYTES , cudaMemcpyHostToDevice);

//	cudaMemcpy(GPU_SORT_TMP_0, SORT_TMP_0, ARRAY_SIZE_IN_BYTES , cudaMemcpyHostToDevice);
//	cudaMemcpy(GPU_SORT_TMP_1, SORT_TMP_1, ARRAY_SIZE_IN_BYTES , cudaMemcpyHostToDevice);


	// radix_sort<<<1, 2>>>(GPU_DATA, NUM_LIST, NUM_ELEM, GPU_SORT_TMP_0, GPU_SORT_TMP_1);

	// radix_sort2<<<1, 1>>>(GPU_DATA, NUM_LIST, NUM_ELEM, GPU_SORT_TMP_1, 0);

	// compute the time

	struct GpuTimer T;
	T.Start();

	gpu_sort_array<<<1, 2>>>(GPU_DATA, NUM_LIST, NUM_ELEM);

	T.Stop();
	float t = T.Elapsed();

	cudaMemcpy(DATA, GPU_DATA, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(GPU_DATA);
//	cudaFree(GPU_SORT_TMP_0);
//	cudaFree(GPU_SORT_TMP_1);

	merge_array(DATA, RESULT, NUM_LIST, NUM_ELEM);

	printf("Before sorting");
	for (u32 i=0; i<ARRAY_SIZE; i++)
	{
		printf("%d ", DATA[i]);
	}

	printf("\nAfter sorting");
	for (u32 i=0; i<ARRAY_SIZE; i++)
	{
		printf("%d ", RESULT[i]);
	}

	printf("\nTime %f (ms)", t);


}
