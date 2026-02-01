#include <iostream>
#include <opencv2/opencv.hpp>

#include "cam_check.hpp"

int main()
{
    cv::VideoCapture cap("/dev/camera0", cv::CAP_V4L2);
    
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;

    for(int i = 0; i < 5; i++) cap.grab();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Empty frame" << std::endl;
            continue; 
        }

        cv::imshow("Camera Check", frame);

        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    return 0;
}