#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{   
    // 1
    // Mat image;
    // cout << "size," << image.size().height << "," << image.size().width << endl;

    // 2
    // Mat M(3,2, CV_8UC3, Scalar(0,0,255));   // BGR
    // cout << "M= " << endl << " " << M << endl;
    // cout << "size," << M.size().height << ","
    //      << M.size().width << endl;

    //3
    // Mat Z = Mat::zeros(2,3, CV_8UC1);
    // cout << "Z = " << endl << " " << Z << endl;

    // Mat O = Mat::ones(2,3, CV_32F);
    // cout << "O = " << endl << " " << O << endl;

    // Mat E = Mat::eye(2,3, CV_64F);
    // cout << "E = " << endl << " " << E << endl;

    //4
    // typedef Vec<uchar, 2> Vec2b;
    // typedef Vec<uchar, 3> Vec3b;
    // typedef Vec<uchar, 4> Vec4b;

    // Vec3b color; // color -> RGB pixel
    // color[0]=255; // B
    // color[1]=0; // G
    // color[2]=0; //R
    // cout << "BRG" << endl << color << endl;

    //5
    // typedef Vec<uchar, 3> Vec3b;
    // Mat grayim(3, 4, CV_8UC1);
    // Mat colorim(3, 4, CV_8UC3);  // BGR

    // //method 1
    // for( int i =0; i < grayim.rows; ++i)
    //     for( int j=0; j < grayim.cols; ++j)
    //         grayim.at<uchar>(i,j) = (i+j)%255;

    // // method2
    // for( int i=0; i < colorim.rows; ++i)
    //     for ( int j=0; j < colorim.cols; ++j)
    //     {
    //         Vec3b pixel;
    //         pixel[0] = i % 255; //Blue
    //         pixel[1] = j % 255; //Green
    //         pixel[2] = 0; // Red
    //         colorim.at<Vec3b>(i,j) = pixel;
    //     }

    // cout << "Gray" << endl << grayim << endl;
    // cout << "BRG" << endl << colorim << endl;
    

    //6
    // typedef Vec<uchar, 3> Vec3b;
    // Mat grayim(3, 4, CV_8UC1);
    // Mat colorim(3, 4, CV_8UC3);  // BGR

    // // interotr
    // MatIterator_<uchar> grayit, grayend;
    // for( grayit=grayim.begin<uchar>(), grayend=grayim.end<uchar>(); grayit!=grayend; ++grayit)
    // {
    //     *grayit = rand() % 255;
    // }

    // MatIterator_<Vec3b> colorit, colorend;
    // for( colorit=colorim.begin<Vec3b>(), colorend=colorim.end<Vec3b>(); colorit!=colorend; ++ colorit)
    // {
    //     (*colorit)[0] = rand() % 255; // Blue
    //     (*colorit)[1] = rand() % 255; // Green
    //     (*colorit)[2] = rand() % 255; // Red
    // }

    
    // cout << "Gray" << endl << grayim << endl;
    // cout << "BRG" << endl << colorim << endl;


    //7 
    //pointer
    typedef Vec<uchar, 3> Vec3b;
    Mat grayim(3, 4, CV_8UC1);
    Mat colorim(3, 4, CV_8UC3);  // BGR

    //method 1
    for( int i =0; i < grayim.rows; ++i)
        // fetch first row pointerr of raw i
        // uchar *p_gray = grayim.ptr<uchar>(i);
        // handle each pixel of row i
        for( int j=0; j < grayim.cols; ++j)
        {
            uchar *p_gray = grayim.ptr<uchar>(i);
            p_gray[j] = (i+j)% 255;
        }
            

    // method2
    for( int i=0; i < colorim.rows; ++i)
        // fetch first row pointerr of raw i
        // Vec3b * p_color = colorim.ptr<Vec3b>(i);
        for ( int j=0; j < colorim.cols; ++j)
        {
            Vec3b * p_color = colorim.ptr<Vec3b>(i);
            p_color[j][0] = i % 255; //Blue
            p_color[j][1] = j % 255; //Green
            p_color[j][2] = 0; // Red

        }

    cout << "Gray" << endl << grayim << endl;
    cout << "BRG" << endl << colorim << endl;

    std::cout << "Image Width: " << colorim.size().width 
        << " Height: " << colorim.size().height 
        << " Channel " << colorim.channels() << std::endl;

    







    return 0;
}