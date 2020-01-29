 #include <stdio.h>

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Number of SM: %d\n", prop.multiProcessorCount);
    printf("  Shared memory of each Thread Block: %f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Maximum Threads of each Thread Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Maximum Threads Per Multi-Processor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
