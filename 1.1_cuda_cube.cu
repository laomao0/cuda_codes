#include <stdio.h>
 
__global__ void cube(float * d_out, float * d_in){
    int tid = threadIdx.x;

    d_out[tid] = d_in[tid] * d_in[tid] * d_in[tid];

}

int main(int argc, char ** arhv){
    const int ARRAY_SIZE = 96;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    //generate the input array on the host
    float h_in[ARRAY_SIZE];
    for( int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] = float(i);
    }

    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float * d_in;
    float * d_out;

    // Allocate GPU Memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    cube<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    //print out result
    for(int i = 0; i < ARRAY_SIZE; i++){
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t": "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;

}