import os
import time
import logging
import threading
import numpy as np
import queue
from queue import Queue

# GObject Introspection for GStreamer
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

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

    def __init__(self, width=640, height=480, fps=15, model_path="/path/to/model.tflite",
                use_model=True, debug_mode=False):
        """
        :param model_path: model (.tflite) or other model file depending on plugin
        :param use_model: use model ai inference or not
        :param debug_mode: show debug frames on screen
        """
        log.info("ImageAnalyticsEngine initialized")
        
        self.width = width
        self.height = height
        self.fps = fps
        self.model_path = model_path
        self.use_model = use_model
        self.debug_mode = debug_mode

        self.pipeline = None
        self.appsink = None
        self.overlay = None

        self._gst_main_loop = GLib.MainLoop()
        self._gst_thread = threading.Thread(target=self._gst_loop, daemon=True)
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
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
            self.overlay = self.pipeline.get_by_name("overlay")
            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")
            if self.overlay is None and self.debug_mode:
                raise RuntimeError("overlay element not found in pipeline")
            
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message::error", self.on_gst_error)
            
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
            log.info("Image Analytics Engine GStreamer thread successfully started")

            # Start main thread
            self._main_thread.start()
            log.info("Image Analytics Engine Main thread successfully started")

    def stop(self):
        """
        Stop the Image Analytics Engine
        """
        log.info("========== STOPPING IMAGE ANALYTICS ENGINE THREAD ==========")

        if self._gst_main_loop.is_running():
            self._gst_main_loop.quit()

        self._gst_thread.join(timeout=3)
        if self._gst_thread.is_alive():
            log.warning("Image Analytics Engine GStreamer thread cannot be stopped")
        else:
            log.info("Image Analytics Engine GStreamer thread successfully stopped")

        self.running = False
        self._stop_event.set()

        self._main_thread.join(timeout=3)
        if self._main_thread.is_alive():
            log.warning("Image Analytics Engine Main thread cannot be stopped")
        else:
            log.info("Image Analytics Engine Main thread successfully stopped")

        log.info("Image Analytics Engine successfully stopped")

    def build_pipeline_str(self):
        r"""
        The pipeline:
                                                       / -> inference ->     \    / -> appsink
          libcamerasrc -> videoconvert -> videoscale ->                        ->
                                                       \ -> non-inference -> /    \ -> autovideosink
        """
        if self.use_model:
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! "
            )
            
            if self.debug_mode:
                # Branch 1: AI inference
                pipeline += (
                    # Split pipeline into two branches
                    f"tee name=t "
                    f"t. ! queue leaky=downstream ! "
                    f"videoscale ! "
                    f"video/x-raw,width=128,height=128,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"tensor_filter framework=tensorflow2-lite model={self.model_path} ! "
                    f"tensor_decoder mode=bounding_boxes option1=blazeface ! "
                    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "   
                )

                # Branch 2: Debug display
                pipeline += (
                    f"t. ! queue leaky=downstream ! videoconvert ! rsvgoverlay name=overlay ! "
                    f"videoconvert ! autovideosink sync=false "
                )
            else:
                pipeline += (
                    f"videoscale ! "
                    f"video/x-raw,width=128,height=128,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"tensor_filter framework=tensorflow2-lite model={self.model_path} ! "
                    f"tensor_decoder mode=bounding_boxes option1=blazeface ! "
                    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "   
                )
        else:
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! "
                # Split pipeline
                f"tee name=t "

                # Branch 1: Raw frames
                f"t. ! queue ! "
                f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "

                # Branch 2: Debug display
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
        # log.info("Received new sample from appsink")
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()

        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            # res_array = np.frombuffer(map_info.data, dtype=np.float32)
            res_array = np.frombuffer(map_info.data, dtype=np.float32).copy()
            log.info("Detection data: %s", res_array)
            
            if len(res_array) > 0:
                ai_results = {"people": True, "fatigue": None, "raw": res_array}
            else:
                ai_results = {"people": False, "fatigue": None, "raw": None}
            
            self.measure_real_fps()
            buf.unmap(map_info)

            if ai_results["people"]:
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(ai_results)

        return Gst.FlowReturn.OK
    
    def on_gst_error(self, bus, message):
        err, debug = message.parse_error()
        log.error(f"GStreamer Error: {err.message}")
        log.error(f"Debug details: {debug}")
    
    # Measure real FPS
    def measure_real_fps(self):
        self.result_count += 1
        if time.time() - self.last_time > 1:
            log.info("Real FPS: %d", self.result_count)
            self.real_fps = self.result_count
            self.result_count = 0
            self.last_time = time.time()

    #  Main loop
    def _main_loop(self):
        while not self._stop_event.is_set():
            result = None
            try:
                result = self.result_queue.get(timeout=1)
            except queue.Empty:
                pass
            except Exception as e:
                log.warning("Error getting result from queue %s", e)

            if self._stop_event.is_set():
                break

            if result is not None:
                raw_data = result.get("raw")
                is_people = result.get("people")

                if is_people and raw_data is not None:
                    # Debug display
                    if self.debug_mode and self.overlay:
                        decoded = decode_blazeface(raw_data, width=self.width, height=self.height)
                        if decoded:
                            x, y, w, h = decoded
                            svg_data = f"""
                            <svg width="{self.width}" height="{self.height}">
                                <rect x="{x}" y="{y}" width="{w}" height="{h}" 
                                style="fill:none;stroke:lime;stroke-width:3" />
                                <text x="{x}" y="{y-10}" fill="lime" font-size="20">FACE DETECTED</text>
                            </svg>
                            """
                        
                            self.overlay.set_property("data", svg_data)
                        else:
                            self.overlay.set_property("data", "")
                    else:
                        log.info("--- [AI DATA] ---")
                        log.info(f"Number of data: {len(raw_data)}")
                        log.info(f"First 5 data: {raw_data[:5]}")
                        log.info("-" * 30)
                else:
                    if self.debug_mode and self.overlay:
                        self.overlay.set_property("data", "")
                    log.info("Waiting for face detection...")

def decode_blazeface(raw_data, score_threshold=0.75, width=640, height=480):
    """
    Decode the array 15,232 into face coordinates. (x, y, w, h)
    """
    if len(raw_data) >= 4:
        x = raw_data[0]
        y = raw_data[1]
        w = raw_data[2]
        h = raw_data[3]
    else:
        return None

    return x, y, w, h

if __name__ == "__main__":
    model_path = "/usr/share/models/blaze_face_short_range.tflite"

    engine = ImageAnalyticsEngine(width=640, height=480, fps=15,
                               model_path=model_path,
                               use_model=True, debug_mode=True)
    try:
        engine.start()
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Interrupted by keyboard")
    except Exception as e:
        log.exception("Unexpected error occurred: %s", e)
    finally:
        engine.stop()