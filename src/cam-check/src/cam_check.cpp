#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>

using namespace libcamera;

/* ===================== Camera Application Class ===================== */

class CameraApp {
public:
    CameraApp(Camera *camera,
              StreamConfiguration &streamConfig,
              CameraManager &cm)
        : camera_(camera),
          streamConfig_(streamConfig),
          cm_(cm)
    {
        cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    }

    void onRequestCompleted(Request *request)
    {
        if (request->status() == Request::RequestCancelled)
            return;

        const FrameBuffer *buffer = request->buffers().begin()->second;
        const FrameBuffer::Plane &plane = buffer->planes()[0];

        void *data = mmap(nullptr,
                          plane.length,
                          PROT_READ,
                          MAP_SHARED,
                          plane.fd.get(),
                          0);

        if (data == MAP_FAILED) {
            std::cerr << "mmap failed" << std::endl;
            return;
        }

        /* YUV420 (I420) */
        cv::Mat yuv(streamConfig_.size.height * 3 / 2,
                    streamConfig_.size.width,
                    CV_8UC1,
                    data);

        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
        cv::imshow("Camera", bgr);

        munmap(data, plane.length);

        if (cv::waitKey(1) == 27) { // ESC
            camera_->stop();
            camera_->release();
            cm_.stop();
            std::exit(0);
        }

        request->reuse(Request::ReuseBuffers);
        camera_->queueRequest(request);
    }

private:
    Camera *camera_;
    StreamConfiguration &streamConfig_;
    CameraManager &cm_;
};

/* =============================== main =============================== */

int main()
{
    CameraManager cm;
    cm.start();

    if (cm.cameras().empty()) {
        std::cerr << "No camera found" << std::endl;
        return -1;
    }

    std::shared_ptr<Camera> camera = cm.cameras()[0];
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config =
        camera->generateConfiguration({ StreamRole::Viewfinder });

    StreamConfiguration &streamConfig = config->at(0);
    streamConfig.pixelFormat = formats::YUV420;
    streamConfig.size.width = 640;
    streamConfig.size.height = 480;

    if (config->validate() == CameraConfiguration::Invalid) {
        std::cerr << "Invalid camera configuration" << std::endl;
        return -1;
    }

    camera->configure(config.get());

    FrameBufferAllocator allocator(camera);
    Stream *stream = streamConfig.stream();

    if (allocator.allocate(stream) < 0) {
        std::cerr << "Failed to allocate buffers" << std::endl;
        return -1;
    }

    std::vector<std::unique_ptr<Request>> requests;
    for (const std::unique_ptr<FrameBuffer> &buffer : allocator.buffers(stream)) {
        std::unique_ptr<Request> request = camera->createRequest();
        request->addBuffer(stream, buffer.get());
        requests.push_back(std::move(request));
    }

    CameraApp app(camera.get(), streamConfig, cm);
    
    camera->requestCompleted.connect(&app,
                                     &CameraApp::onRequestCompleted);

    camera->start();

    for (auto &request : requests)
        camera->queueRequest(request.get());

    /* Main loop */
    while (true)
        usleep(1000);

    return 0;
}
