#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

#define DEBUG (1)

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);  // notice we convert BGR to RGB

  //allocate memory for the output, the gray scale image
  // 8U-char 0-255, C1-one channel
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  if DEBUG
  {
    unsigned char *p_gray = imageGrey.ptr<unsigned char>(0);
    for (int i=0; i<1;i++)
    {
      std::cout << "p_gray[i]" << int(p_gray[i]) << std::endl;
    }
    
    
    for (int i=0; i<1;i++)
    {
      std::cout << "*greyImage[i]" << int( (*greyImage)[i]) << std::endl;
    }

    for (int i=0; i<1;i++){
      std::cout << "*inputImage[i]" << ((*inputImage)[i].x) << std::endl;
    }
      
  }

  const size_t numPixels = numRows() * numCols();

  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  // cudaMemset set all to zero
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

  if DEBUG
    std::cout << "Inital cuda Memory " << std::endl;
    std::cout << "Total number of pixels " << numPixels << std::endl;

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

  //output the image
  cv::imwrite(output_file.c_str(), output);
}

void cleanup()
{
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}
