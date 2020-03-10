// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.
#define BLOCKDIMX (32)
#define BLOCKDIMY (16)
#define CHANNEL (1)
#include "utils.h"
#include <iostream>


__global__
void rbga_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset

  const int DIM_GRID_X = gridDim.x;
  const int DIM_GRID_Y = gridDim.y;
  const int DIM_BLOCK_X = blockDim.x;
  const int DIM_BLOCK_Y = blockDim.y;

  const int w_i = blockIdx.x * DIM_BLOCK_X + threadIdx.x;
  const int h_i = blockIdx.y * DIM_BLOCK_Y + threadIdx.y;

  const bool w_bound = w_i < numCols;
  const bool h_bound = h_i < numRows;

  int offset = h_i * numCols + w_i;

  if ( w_bound && h_bound){

    uchar4 data4 = rgbaImage[offset];

    float R = data4.x;
    float G = data4.y;
    float B = data4.z;

    float greyness = .299f * R + .587f * G + .114f * B;

    greyImage[offset] = static_cast<unsigned char>(greyness);  // float to uchar

  }
  
  
}



void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(BLOCKDIMX, BLOCKDIMY, 1);  //TODO
  const dim3 gridSize( (numCols + BLOCKDIMX -1) / BLOCKDIMX, (numRows + BLOCKDIMY -1) / BLOCKDIMY, CHANNEL );  //TODO

  std::cout << "blockSize " << BLOCKDIMX << ","<< BLOCKDIMY << ",1" << std::endl;
  std::cout << "gridSize " << (numCols + BLOCKDIMX -1) / BLOCKDIMX << ","
                           << (numRows + BLOCKDIMY -1) / BLOCKDIMY << ","
                           << CHANNEL << std::endl;
 
  rbga_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

}
