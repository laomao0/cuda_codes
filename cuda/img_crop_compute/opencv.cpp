#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
using namespace std;
using namespace cv;

int operateMat(Mat img, int startx, int starty, int size, int thresh);

int main()
{
	Mat img = imread("/DATA/wangshen_data/CODES/cuda_codes/cuda/img_crop_compute/img/test.jpg",0);
	int size = 40;
	int thresh = 0;
	// cout << "Please Input Size:";
	// cin >> size;   //分块区域大小
	// cout << "Please Input Thresh value:";
	// cin >> thresh; //阈值
	double time0 = static_cast<double>(getTickCount()); //计时器开始
	int row = img.rows;
	int col = img.cols;
	int length = row / size;
	int width = col / size;
	int sumResult[10000];     //区域内求和结果
	int averageResult[10000]; //区域内均值结果
	int maxResult[10000]; //区域内最大值结果
	int minResult[10000]; //区域内最小值结果
	int threshNumber[10000];  //区域内阈值结果

	int count = 0;
	int x = 0;

	for (int k = 0; k < 1000; k++)
	{	
		count = 0;
		x = 0;
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < width; j++)
			{
				sumResult[count], maxResult[count], minResult[count], threshNumber[count]  
				    = operateMat(img, i*size, j*size, size, 35);
				averageResult[count] = sumResult[count] / (size * size);
				count += 1;
			}
		}
	}
	time0 = ((double)getTickCount() - time0) / getTickFrequency(); //计时器结束
	cout << time0 << "ms" << endl; //输出运行时间
	system("Pause");
	return 0;
}

int operateMat(Mat img, int startx, int starty, int size, int thresh)
{
	int sum = 0;
	int max = 0;
	int min = 255;
	int average = 0;
	int threshCount = 0;
	for (int i = startx; i < startx + size; i++)
	{
		uchar *data = img.ptr<uchar>(i);
		for (int j = starty; j < starty + size; j++)
		{
			sum += data[j];
			if (max < data[j])
			{
				max = data[j];
			}
			if (min > data[j])
			{
				min = data[j];
			}
			if (data[j] > thresh)
			{
				threshCount += 1;
			}
		}
	}
	average = sum / (size*size);
	return sum, max, min, threshCount;
}