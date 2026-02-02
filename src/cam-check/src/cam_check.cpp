#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open /dev/video0 via V4L2" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;

    if (!cap.read(frame)) {
        std::cerr << "ERROR: Failed to read frame" << std::endl;
        return -1;
    }

    std::cout << "Camera OK. Frame size: "
              << frame.cols << "x" << frame.rows
              << " Channels: " << frame.channels()
              << std::endl;

    return 0;
}