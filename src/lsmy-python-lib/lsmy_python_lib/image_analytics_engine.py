import os
import cv2
import time
import logging
import threading

log = logging.getLogger("image-analytics-engine")

# Setup wayland environment variables
os.environ["XDG_RUNTIME_DIR"] = "/run/user/0"
os.environ["WAYLAND_DISPLAY"] = "wayland-0"

class ImageAnalyticsEngine:
    """
    ImageAnalyticsEngine is a class that handles the camera and AI logic.
    It uses the GStreamer pipeline to capture frames from the camera and passes them to the AI model."""

    def __init__(self, width=640, height=480, callback=None):
        """
        :param callback: The function that will be called whenever there is a new AI result
        """
        log.info("ImageAnalyticsEngine initialized")
        self.width = width
        self.height = height
        # Place to send results externally (CoreIoT, Log, etc.)
        self.callback = callback
        
        # Pipeline GStreamer for camera
        self.pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={self.width}, height={self.height}, framerate=15/1 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
        )
        
        self.cap = None
        self.latest_frame = None

        self.running = False
        self._stop_event = threading.Event()
        self.image_analytics_engine_thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        """
        Start the Image Analytics Engine
        """
        if not self.running:
            self.running = True
            self._stop_event.clear()

            self.image_analytics_engine_thread.start()
            log.info("Image Analytics Engine thread successfully started")

    def stop(self):
        """
        Stop the Image Analytics Engine
        """
        log.info("========== STOPPING IMAGE ANALYTICS ENGINE THREAD ==========")
        self.running = False
        self._stop_event.set()

        self.image_analytics_engine_thread.join(timeout=3)
        if self.image_analytics_engine_thread.is_alive():
            log.warning("Image Analytics Engine thread cannot be stopped")
        else:
            log.info("Image Analytics Engine thread successfully stopped")

        # Clean actions
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

    def _run(self):
        log.info("========== STARTING IMAGE ANALYTICS ENGINE THREAD ==========")
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            log.error("Cannot open GStreamer Pipeline!")
            return

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                # 1. Preprocessing (Resize, Normalize, etc.)
                # blob = cv2.resize(frame, (300, 300))

                # 2. Run AI Model Inference Logic
                ai_results = self._ai_inference_logic(frame)

                self.latest_frame = frame

                # 3. Postprocessing - Call the callback function
                if self.callback:
                    self.callback(ai_results, frame)

                # self._stop_event.wait(3)
            except Exception as e:
                log.error(f"Error in Image Analytics Engine: {e}")

    def _ai_inference_logic(self, frame):
        """
        AI Inference Logic
        """
        # Run AI Model Inference
        time.sleep(0.05)
        person_count = 2 
        is_fatigued = False
        
        return {"people": person_count, "fatigue": is_fatigued}

# Callback function
def my_iot_logic(results, frame):
    """
    This function will be called whenever there is a new AI result
    """
    # log.info(f"Received AI results: {results['people']} people in the room.")
    
    # Display the frame to debug
    cv2.imshow("LSMY Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        log.info("Exit key pressed")

if __name__ == "__main__":
    def silent_callback(results, frame):
        pass

    my_camera = ImageAnalyticsEngine(callback=silent_callback)
    my_camera.start()

    try:
        while True:
            # Do something
            frame = my_camera.latest_frame
            if frame is not None:
                cv2.imshow("LSMY Monitor", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info("Exit key pressed")
                break
            
            time.sleep(0.01)
            # time.sleep(1)
    except KeyboardInterrupt:
        my_camera.stop()