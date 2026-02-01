#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::string pipeline = "v4l2src device=/dev/video0 ! "
                           "video/x-bayer,format=bggr,width=640,height=480 ! "
                           "bcm2835isp ! "
                           "video/x-raw,format=BGR ! "
                           "videoconvert ! appsink drop=true";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera via GStreamer!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("LSMY Camera Check", frame);
        if (cv::waitKey(30) == 27) break;
    }
    return 0;
}