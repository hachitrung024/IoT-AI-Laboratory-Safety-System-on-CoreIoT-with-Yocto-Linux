import os
import time
import psutil
import logging
import argparse
import threading
import subprocess
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
        self.infer_start = None
        self.infer_end = None

        self._infer_timestamps = {}

        self._gst_main_loop = GLib.MainLoop()
        self._gst_thread = threading.Thread(target=self._gst_loop, daemon=True)
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._stop_event = threading.Event()

        self.metrics_lock = threading.Lock()
        self.critical_lock = threading.Lock()
        self.draw_overlay_lock = threading.Lock()

        self.pipeline_fps = fps
        self.result_count = 0
        self.last_time = time.time()

        self.avg_inference_time = 0.0
        self.avg_pipeline_latency = 0.0
        self.inference_time = 0.0
        self.pipeline_latency = 0.0

        self.overlay_update_time = 0

        self.current_bbox = None
        self.current_landmarks = None

        self.result_queue = Queue(maxsize=1)

        self.running = False
        self.is_critical = False

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

            self.infer_start = self.pipeline.get_by_name("infer_start")
            self.infer_end = self.pipeline.get_by_name("infer_end")

            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")
            if self.overlay is None and self.debug_mode:
                raise RuntimeError("overlay element not found in pipeline")
            if self.infer_start is None and self.use_model:
                raise RuntimeError("infer_start element not found in pipeline")
            if self.infer_end is None and self.use_model:
                raise RuntimeError("infer_end element not found in pipeline")
            
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message::error", self.on_gst_error)
            
            # Configure appsink
            # emit-signals true to connect to "new-sample" signal
            self.appsink.set_property("emit-signals", True)
            self.appsink.set_property("sync", False)

            # Connect signal: new-sample
            self.appsink.connect("new-sample", self.on_new_sample)

            # Connect signal: AI inference
            if self.use_model:
                self.infer_start.connect("handoff", self.on_infer_start)
                self.infer_end.connect("handoff", self.on_infer_end)
            if self.debug_mode:
                self.overlay.connect("draw", self.on_draw_overlay)

            # Start pipeline in a dedicated thread with GLib MainLoop
            self.running = True
            self.is_critical = False
            self._stop_event.clear()

            # Start GStreamer thread
            self._gst_thread.start()
            log.info("Image Analytics Engine GStreamer thread successfully started")

            # Start main thread
            self._main_thread.start()
            log.info("Image Analytics Engine Main thread successfully started")

            # Start monitor thread
            self._monitor_thread.start()
            log.info("Image Analytics Engine Monitor thread successfully started")

            log.info("Image Analytics Engine successfully started")

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

        self._stop_event.set()

        self._main_thread.join(timeout=3)
        if self._main_thread.is_alive():
            log.warning("Image Analytics Engine Main thread cannot be stopped")
        else:
            log.info("Image Analytics Engine Main thread successfully stopped")

        self._monitor_thread.join(timeout=3)
        if self._monitor_thread.is_alive():
            log.warning("Image Analytics Engine Monitor thread cannot be stopped")
        else:
            log.info("Image Analytics Engine Monitor thread successfully stopped")

        self.running = False
        self.is_critical = False
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
                f"tee name=t "
            )
            
            if self.debug_mode:
                pipeline += (
                    # Crop tensor
                    f"tensor_mux name=mux ! "
                    f"other/tensors ! "
                    f"tensor_crop ! "
                    f"tensor_decoder mode=direct_video ! video/x-raw ! "

                    # Face landmark detection
                    f"videoscale ! video/x-raw,width=192,height=192 ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"tensor_filter framework=tensorflow2-lite model=/usr/share/models/face_landmark.tflite custom=delegate:xnnpack ! "

                    # Decode + Ear detection
                    f"tensor_filter framework=face_mesh_decode model=dummy1 custom={self.width},{self.height} ! "
                    # f"tensor_filter framework=ear_eval model=dummy2 ! "

                    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "  
                )

                # Branch 1: Frame raw branch
                pipeline += (
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"queue ! mux.sink_0 "
                )

                # Branch 2: Face detection Model
                pipeline += (
                    # Split pipeline into three branches
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoscale ! "
                    f"video/x-raw,width=128,height=128 ! "
                    f"videoconvert ! "
                    f"video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"identity name=infer_start signal-handoffs=true ! "
                    f"tensor_filter framework=tensorflow2-lite model={self.model_path} custom=delegate:xnnpack ! "
                    f"tensor_filter framework=blaze_decode model=dummy custom={self.width},{self.height} ! "
                    f"identity name=infer_end signal-handoffs=true ! "
                    f"tensor_transform mode=arithmetic option=typecast:int32,add:0 ! "
                    f"queue ! mux.sink_1 "
                )

                # Branch 3: Debug display
                pipeline += (
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! cairooverlay name=overlay ! "
                    f"autovideosink sync=false "
                )
            else:
                pipeline += (
                    f"videoscale ! "
                    f"video/x-raw,width=128,height=128 ! "
                    f"videoconvert ! "
                    f"video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"identity name=infer_start signal-handoffs=true ! "
                    f"tensor_filter framework=tensorflow2-lite model={self.model_path} custom=delegate:xnnpack ! "
                    # f"tensor_mux name=mux ! "
                    f"tensor_filter framework=blaze_decode model=dummy custom={self.width},{self.height} ! "
                    f"identity name=infer_end signal-handoffs=true ! "
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
        
        clock = self.pipeline.get_clock()
        base_time = self.pipeline.get_base_time()

        pipeline_latency = 0
        if clock and buf.pts != Gst.CLOCK_TIME_NONE:
            current_pipeline_time = clock.get_time() - base_time
            pipeline_latency = (current_pipeline_time - buf.pts) / Gst.MSECOND

        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                res_array = np.frombuffer(map_info.data, dtype=np.float32).copy()
                
                if len(res_array) == 936:
                    ai_results = {"people": True, "fatigue": None, "raw": res_array}
                else:
                    ai_results = {"people": False, "fatigue": None, "raw": None}
                
                self.measure_pipeline_metrics(self.inference_time, pipeline_latency)

                if ai_results["people"]:
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                    self.result_queue.put(ai_results)
            except Exception as e:
                log.error("Error processing inference result: %s", e)
            finally: 
                buf.unmap(map_info)

        return Gst.FlowReturn.OK
    
    def on_gst_error(self, bus, message):
        err, debug = message.parse_error()
        log.error(f"GStreamer Error: {err.message}")
        log.error(f"Debug details: {debug}")

    def on_infer_start(self, element, buffer):
        if buffer.pts != Gst.CLOCK_TIME_NONE:
            self._infer_timestamps[buffer.pts] = time.time()

    def on_infer_end(self, element, buffer):
        start = self._infer_timestamps.pop(buffer.pts, None)
        if start:
            self.inference_time = (time.time() - start) * 1000

    def on_draw_overlay(self, overlay, context, timestamp, duration):
        # with self.draw_overlay_lock:
        #     if self.current_bbox is None:
        #         return

        #     x, y, w, h = self.current_bbox

        # draw bounding box
        # context.set_source_rgb(0, 1, 0)
        # context.set_line_width(3)
        # context.rectangle(x, y, w, h)
        # context.stroke()

        with self.draw_overlay_lock:
            # draw landmarks
            if self.current_landmarks:
                context.set_source_rgb(1, 0, 0)
                for kx, ky in self.current_landmarks:
                    context.arc(kx, ky, 3, 0, 2 * 3.1416)
                    context.fill()

        # draw info panel
        context.set_source_rgba(0, 0, 0, 0.5)
        context.rectangle(5, 5, 250, 75)
        context.fill()

        context.set_source_rgb(1, 1, 1)
        context.select_font_face("monospace", 0, 0)
        context.set_font_size(14)

        context.move_to(15, 25)
        context.show_text(f"FPS: {self.pipeline_fps}")

        context.move_to(15, 45)
        context.show_text(f"AI Latency: {self.avg_inference_time:.2f} ms")

        context.move_to(15, 65)
        context.show_text(f"Pipeline Latency: {self.avg_pipeline_latency:.2f} ms")
    
    # Measure pipeline FPS
    def measure_pipeline_metrics(self, inference_time=0, pipeline_latency=0):
        # Pipeline FPS
        self.result_count += 1
        if time.time() - self.last_time > 1:
            with self.metrics_lock:
                self.pipeline_fps = self.result_count
            self.result_count = 0
            self.last_time = time.time()

        with self.metrics_lock:
            # Inference time
            if inference_time > 0:
                if self.avg_inference_time == 0:
                    self.avg_inference_time = inference_time
                else:
                    self.avg_inference_time = (self.avg_inference_time * 0.9) + (inference_time * 0.1)

            # Pipeline latency
            if pipeline_latency > 0:
                if self.avg_pipeline_latency == 0:
                    self.avg_pipeline_latency = pipeline_latency
                else:
                    self.avg_pipeline_latency = (self.avg_pipeline_latency * 0.9) + (pipeline_latency * 0.1)

    def evaluate_pipeline_status(self):
        # --- Metrics ---
        # 1. CPU Temp
        try:
            temp = psutil.sensors_temperatures()['cpu_thermal'][0].current
        except:
            temp = 0
        
        # 2. CPU Usage
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # 3. RAM Usage
        ram = psutil.virtual_memory()

        # --- Evaluations ---
        is_critical = False
        status_msg = []
        if temp > 80:
            is_critical = True
            status_msg.append(f"CRITICAL TEMP: {temp}°C")
        elif temp > 70:
            log.warning(f"High Temperature Warning: {temp}°C")

        if cpu_usage > 95:
            is_critical = True
            status_msg.append(f"CPU OVERLOAD: {cpu_usage}%")
        elif cpu_usage > 85:
            log.warning(f"High CPU Usage: {cpu_usage}% - Pipeline might lag.")

        if ram.percent > 90:
            is_critical = True
            status_msg.append(f"LOW MEMORY: {ram.percent}%")

        if is_critical:
            try:
                clock_raw = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode().strip()
                clock_mhz = int(clock_raw.split('=')[1]) / 1_000_000
                
                volts = subprocess.check_output(["vcgencmd", "measure_volts", "core"]).decode().strip()
                
                # Throttled Status
                # 0x0: Normal
                # 0x50000: Previously throttled due to overheating
                # 0x50005: Currently throttled and power supply is insufficient
                throttled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()

                if is_critical:
                    log.error(f"DIAGNOSTICS: Clock: {clock_mhz}MHz | {volts} | Status: {throttled}")
                    if "0x" in throttled and throttled != "throttled=0x0":
                        log.error("SYSTEM ALERT: Hardware throttling detected! Check Power Supply or Cooling.")
                        with self.critical_lock:
                            self.is_critical = True
                        return is_critical
            except Exception as e:
                log.debug(f"Could not run vcgencmd: {e}")

        # Print status
        log.info(f"--- PIPELINE STATUS ---")
        log.info(f"CPU: {cpu_usage}% | Temp: {temp}°C | RAM: {ram.percent}%")
        with self.metrics_lock:
            log.info(f"Camera FPS: {self.fps} | Pipeline FPS: {self.pipeline_fps}")
            log.info(f"AI Latency: {self.avg_inference_time:.2f}ms | Pipeline Latency: {self.avg_pipeline_latency:.2f}ms")
        if is_critical:
            log.error(f"CRITICAL STATUS: {' | '.join(status_msg)}")
        log.info("-" * 30)

        return is_critical

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

                now = time.time()
                if now - self.overlay_update_time > 0.1:
                    self.overlay_update_time = now
                    if is_people and raw_data is not None:
                        # Debug display
                        if self.debug_mode and self.overlay:
                            with self.draw_overlay_lock:
                                self.current_bbox = None

                                # reshape landmarks
                                lm = raw_data.reshape(-1, 2)  # (468, 2)
                                self.current_landmarks = lm
                        else:
                            # log.info("--- [AI DATA] ---")
                            # log.info(f"Number of data: {len(raw_data)}")
                            # log.info(f"First 5 data: {raw_data[:5]}")
                            # log.info("-" * 30)
                            pass
                    else:
                        with self.draw_overlay_lock:
                            self.current_bbox = None
                            self.current_landmarks = None

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            is_critical = self.evaluate_pipeline_status()
            
            if is_critical:
                break
            
            self._stop_event.wait(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Analytics Engine")

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Camera FPS (default: 15)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed logging and visualization"
    )

    args = parser.parse_args()

    model_path = "/usr/share/models/blaze_face_short_range.tflite"

    engine = ImageAnalyticsEngine(width=640, height=480, fps=args.fps,
                               model_path=model_path,
                               use_model=True, debug_mode=args.debug)
    try:
        engine.start()
        
        while True:
            with engine.critical_lock:
                if engine.is_critical:
                    log.info("Critical status detected, stopping engine...")
                    engine.stop()
                    break
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("Interrupted by keyboard")
    except Exception as e:
        log.exception("Unexpected error occurred: %s", e)
    finally:
        if engine.running:
            engine.stop()
        else:
            log.info("Skipping stop as engine is not running")