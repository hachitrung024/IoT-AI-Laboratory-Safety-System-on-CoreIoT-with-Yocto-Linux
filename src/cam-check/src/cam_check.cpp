#include <iostream>
#include <opencv2/opencv.hpp>

#include "cam_check.hpp"

int main()
{
    // Camera /dev/video0
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Cannot open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "ERROR: Empty frame" << std::endl;
            break;
        }

        cv::imshow("Camera Check", frame);

        // ESC key to exit
        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
}
