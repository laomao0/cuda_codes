#include <stdio.h>
#define NUM_BLOCKS (1)
#define BLOCK_WIDTH (4)

__global__ 
void hello()
{

    int idx = threadIdx.x;

    // all the threads in the block can access the block-shared memory
    __shared__ int array[BLOCK_WIDTH];  

    // initize the memory
    array[idx] = threadIdx.x;

    // stop all the threads to read before initilized completely
    printf("Hello world! I'm a thread in block %d\n", idx);
    __syncthreads();
    printf("Hello world! I'm a thread in block %d\n", idx);

    if (idx < BLOCK_WIDTH-1)
    {
        // pre-read the memory
        int temp = array[idx+1];   
        __syncthreads();

        // when read over, write it
        array[idx] = temp;
        __syncthreads(); // use sync to assure all the memory have written
        
    }

}

int main()
{   
    hello<<<NUM_BLOCKS, BLOCK_WIDTH>>>();

    // force the printf()s to flush
    cudaDeviceSynchronize();

    printf("That's all!\n");

    return 0;
}