#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

#define DEBUG (0)

void compareImages(std::string reference_filename, std::string test_filename, 
                   bool useEpsCheck, double perPixelError, double globalError)
{
  cv::Mat reference = cv::imread(reference_filename, -1); // read flag=-1, IMREAD_UNCHANGED
  cv::Mat test = cv::imread(test_filename, -1);

  if DEBUG
    std::cout << "Image Width: " << reference.size().width 
              << " Height: " << reference.size().height 
              << " Channel " << reference.channels() << std::endl;

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  if DEBUG
    std::cout << "diffSingleChannel Width: " << diffSingleChannel.size().width << " Height: " << diffSingleChannel.size().height << std::endl;

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  if DEBUG
    std::cout << "new diffSingleChannel Width: " << diff.cols << " Height: " << diff.rows << std::endl;

  cv::imwrite("HW1_differenceImage.png", diff);
  //OK, now we can start comparing values...
  // fetch the first row pointer 
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) {
    checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
  }
  else
  {
    checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}


// int main()
// {
//   std::string reference_filename = "cinque_terre_small.jpg";
//   std::string test_filename = "cinque_terre_small.jpg";

//   compareImages(reference_filename, test_filename, 1, 0.0, 0.0);

//   return 0;

// }
