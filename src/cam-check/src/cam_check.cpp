#include <iostream>
#include <opencv2/opencv.hpp>

#include "cam_check.hpp"

int main()
{
    cv::VideoCapture cap;
    if (!cap.open("/dev/video0", cv::CAP_V4L2)) {
        std::cerr << "ERROR: Cannot open camera" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc('B','G','R','3'));

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Empty frame" << std::endl;
            break;
        }

        cv::imshow("Camera Check", frame);

        if (cv::waitKey(30) == 27)
            break;
    }

    return 0;
}
