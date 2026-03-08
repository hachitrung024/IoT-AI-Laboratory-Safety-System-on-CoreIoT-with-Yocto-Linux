import os
import time
import logging
import threading
import numpy as np
from queue import Queue

# For showing frames to debug
import cv2

# GObject Introspection for GStreamer
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

log = logging.getLogger("image-analytics-engine")

# Initialize GStreamer Library
Gst.init(None)

# Setup wayland environment variables
os.environ["XDG_RUNTIME_DIR"] = "/run/user/0"
os.environ["WAYLAND_DISPLAY"] = "wayland-0"

class ImageAnalyticsEngine:
    """
    ImageAnalyticsEngine is a class that handles the camera and AI logic.
    It uses the GStreamer pipeline to capture frames from the camera and passes them to the AI model.
    It also handles the AI inference logic and the callback function.
    """

    def __init__(self, width=640, height=480, fps=15, model_path="/path/to/model.xml",
                 model_proc="/path/to/model-proc.json", use_model=True):
        """
        :param model_path: model (OpenVINO .xml) or other model file depending on plugin
        :param model_proc: model-proc for GVA (optional)
        :param use_model: use model ai inference or not
        """
        log.info("ImageAnalyticsEngine initialized")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.model_path = model_path
        self.model_proc = model_proc
        self.use_model = use_model

        self.pipeline = None
        self.appsink = None

        self._gst_main_loop = GObject.MainLoop()
        self._gst_thread = threading.Thread(target=self._gst_loop, daemon=True)
        self._stop_event = threading.Event()

        self.real_fps = fps
        self.result_count = 0
        self.last_time = time.time()

        self.result_queue = Queue(maxsize=1)

        self.running = False

    def start(self):
        """
        Start the Image Analytics Engine
        """
        log.info("========== STARTING IMAGE ANALYTICS ENGINE THREAD ==========")

        if not self.running:
            # Start the Image Analytics Engine Pipeline
            if self.pipeline is not None:
                log.warning("Engine already started")
                return

            pipeline_str = self.build_pipeline_str()
            log.info("Creating pipeline: %s", pipeline_str)

            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")
            
            # Configure appsink
            # emit-signals true to connect to "new-sample" signal
            self.appsink.set_property("emit-signals", True)
            self.appsink.set_property("sync", False)

            # Connect signal: new-sample
            self.appsink.connect("new-sample", self.on_new_sample)

            # Start pipeline in a dedicated thread with GLib MainLoop
            self.running = True
            self._stop_event.clear()
            self._gst_thread.start()
            log.info("Image Analytics Engine thread successfully started")

            # Start main loop
            self.main_loop()

    def stop(self):
        """
        Stop the Image Analytics Engine
        """
        log.info("========== STOPPING IMAGE ANALYTICS ENGINE THREAD ==========")

        if self._gst_main_loop.is_running():
            self._gst_main_loop.quit()

        self._gst_thread.join(timeout=3)
        if self._gst_thread.is_alive():
            log.warning("Image Analytics Engine thread cannot be stopped")
        else:
            log.info("Image Analytics Engine thread successfully stopped")

        self.running = False
        self._stop_event.set()

        cv2.destroyAllWindows()

        log.info("Image Analytics Engine successfully stopped")

    def build_pipeline_str(self):
        """
        The pipeline:
                                                       / -> inference ->     \    / -> appsink
          libcamerasrc -> videoconvert -> videoscale ->                        ->
                                                       \ -> non-inference -> /    \ -> autovideosink
        """
        if self.use_model:
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! videoscale ! "

                # AI inference
                f"gvainference model={self.model_path} "
                f"model-proc={self.model_proc} device=CPU ! "

                # Split pipeline
                f"tee name=t "

                # Branch 1: python metadata
                f"t. ! queue ! "
                f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "

                # Branch 2: debug display
                f"t. ! queue ! "
                f"gvawatermark ! "
                f"autovideosink sync=false"
            )

        else:
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! "
                f"tee name=t "

                f"t. ! queue ! "
                f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "

                f"t. ! queue ! "
                f"autovideosink sync=false"
            )
        return pipeline
    
    def _gst_loop(self):
        """
        Main loop of the GStreamer thread
        """
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self._gst_main_loop.run()
        except Exception as e:
            log.exception("GStreamer main loop stopped with exception: %s", e)
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            log.info("Pipeline stopped")

    # --------------------
    # appsink handler
    # --------------------
    def on_new_sample(self, appsink):
        """
        Called in GStreamer thread context when appsink has a new sample.
        Convert sample -> numpy array and extract metadata.
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()

        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        fmt = structure.get_value('format')

        # ai_results = parse_ai_metadata(buf)
        ai_results = {"people": None, "fatigue": None, "raw": None}
        self.measure_real_fps()

        # Push results to queue
        if all(v is not None for v in ai_results.values()):
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass
            self.result_queue.put(ai_results)

        return Gst.FlowReturn.OK
    
    # Measure real FPS
    def measure_real_fps(self):
        self.result_count += 1
        if time.time() - self.last_time > 1:
            log.info("Real FPS: %d", self.result_count)
            self.real_fps = self.result_count
            self.result_count = 0
            self.last_time = time.time()

    #  Main loop
    def main_loop(self):
        while not self._stop_event.is_set():
            result = self.result_queue.get()
            if result is not None:
                # Postprocessing results
                pass

if __name__ == "__main__":
    model_xml = "/opt/models/people_counter/FP32/model.xml"
    model_proc = "/opt/models/people_counter/model-proc.json"

    engine = ImageAnalyticsEngine(width=640, height=480, fps=15,
                               model_path=model_xml, model_proc=model_proc,
                               use_model=False)
    try:
        engine.start()
    except KeyboardInterrupt:
        log.info("Interrupted by keyboard")
    finally:
        engine.stop()