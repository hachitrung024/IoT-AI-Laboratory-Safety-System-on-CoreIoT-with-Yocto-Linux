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

# Model blaze face detection path
MODEL_BLAZE_FACE_DETECTION = "/usr/share/models/blaze_face_short_range.tflite"

# Model landmark path
MODEL_LANDMARK_FACE_DETECTION = "/usr/share/models/face_landmark.tflite"

# Landmark input size model 192x192
LANDMARK_INPUT_SIZE = 192

# Fatigue states
F_STATE_NORMAL = 0
F_STATE_WARNING = 1
F_STATE_TIRED = 2
F_STATE_DISTRACTED = 3
F_STATE_NO_FACE = 4

class ImageAnalyticsEngine:
    """
    ImageAnalyticsEngine is a class that handles the camera and AI logic.
    It uses the GStreamer pipeline to capture frames from the camera and passes them to the AI model.
    It also handles the AI inference logic and the callback function.
    """

    def __init__(self, width=640, height=480, fps=15, model_blaze_path="/path/to/model.tflite", model_landmark_path="/path/to/landmark_model.tflite",
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
        self.model_blaze_path = model_blaze_path
        self.model_landmark_path = model_landmark_path
        self.use_model = use_model
        self.debug_mode = debug_mode

        self.output_crop_dim = self.width * self.height

        self.num_landmarks = 468
        self.landmarks_dim = 2  # (x, y) we not using z for 3D
        self.output_landmarks_dim = self.num_landmarks * self.landmarks_dim
        self.output_fatigue_dim = 14

        self.pipeline = None
        self.appsink = None
        self.cropsink = None
        self.bboxsink = None
        self.landmarksink = None

        self.overlay = None
        
        self.infer_start = None
        self.infer_end = None

        self._infer_timestamps = {}

        self._gst_main_loop = GLib.MainLoop()
        self._gst_thread = None
        self._main_thread = None
        self._monitor_thread = None
        self._stop_event = threading.Event()

        self.metrics_lock = threading.Lock()
        self.critical_lock = threading.Lock()
        self.draw_overlay_lock = threading.Lock()
        self.fatigue_lock = threading.Lock()

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

        self.current_fatigue_output = None
        self.current_state = F_STATE_NO_FACE

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
            # self.cropsink = self.pipeline.get_by_name("cropsink")
            self.bboxsink = self.pipeline.get_by_name("bboxsink")
            self.landmarksink = self.pipeline.get_by_name("landmarksink")

            self.overlay = self.pipeline.get_by_name("overlay")

            self.infer_start = self.pipeline.get_by_name("infer_start")
            self.infer_end = self.pipeline.get_by_name("infer_end")

            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")
            # if self.cropsink is None and self.debug_mode:
            #     raise RuntimeError("cropsink element not found in pipeline")
            if self.bboxsink is None and self.debug_mode:
                raise RuntimeError("bboxsink element not found in pipeline")
            if self.landmarksink is None and self.debug_mode:
                raise RuntimeError("landmarksink element not found in pipeline")
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
                # self.cropsink.connect("new-data", self.on_new_crop_debug)
                self.bboxsink.connect("new-data", self.on_new_bbox)
                self.landmarksink.connect("new-data", self.on_new_landmarks)

            # Start pipeline in a dedicated thread with GLib MainLoop
            self.running = True
            self.is_critical = False
            self._stop_event.clear()

            self._gst_thread = threading.Thread(target=self._gst_loop, daemon=True)
            self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)

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
                                                       / -> Raw frame ->     \    
          libcamerasrc -> videoconvert -> videoscale ->                        -> tensor_crop -> Face mesh -> /
                                                       \ -> Face detection -> /  
                                                        \ -> Overlay -> /
        """

        if self.use_model:
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"tee name=t "
            )

            if self.debug_mode:
                # Two sinks for tensor crop:
                # 1. Raw frame 
                # 2. Face detection result
                pipeline += (
                    f"tensor_crop name=crop silent=false "
                )

                # 1. Raw frame
                pipeline += (
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"crop.raw "
                )

                # 2. Face detection result
                pipeline += (
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoscale ! video/x-raw,width=128,height=128 ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"identity name=infer_start signal-handoffs=true ! "

                    # Blaze face detection model
                    f"tensor_filter framework=tensorflow2-lite "
                    f"model={self.model_blaze_path} custom=delegate:xnnpack ! "

                    # Blaze decode plugin
                    f"tensor_filter framework=blaze_decode model=dummy "
                    f"custom={self.width},{self.height} ! "
                    f"tee name=td "
                    f"td. ! queue max-size-buffers=2 leaky=downstream ! tensor_sink name=bboxsink "

                    f"td. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"identity name=infer_end signal-handoffs=true ! "
                    f"crop.info "
                )

                # Merge tensor crop pipeline
                pipeline += (
                    f"crop. ! "

                    # --- Debug crop ---
                    # f"tensor_debug name=debug_crop ! "
                    # f"tensor_sink name=cropsink"
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"crop_decode ! "

                    # --- Debug crop view ---
                    # f"crop_view ! videoconvert ! autovideosink sync=false"

                    # Face landmark detection
                    f"tensor_filter framework=tensorflow2-lite model={self.model_landmark_path} custom=delegate:xnnpack ! "

                    # Decode + Ear detection
                    f"tensor_filter framework=face_mesh_decode model=dummy1 custom={self.width},{self.height} ! "
                    f"tee name=tf "
                    f"tf. ! queue max-size-buffers=2 leaky=downstream ! tensor_sink name=landmarksink "

                    f"tf. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"tensor_filter framework=fatigue_eval model=dummy3 ! "

                    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "  
                )

                # 3. Debug display
                pipeline += (
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! cairooverlay name=overlay ! "
                    f"autovideosink sync=false "
                )
            else:
                # No debug
                pipeline += (
                    f"tensor_crop name=crop silent=false "

                    # Raw frame
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"crop.raw "

                    # Face detection
                    f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                    f"videoscale ! video/x-raw,width=128,height=128 ! "
                    f"videoconvert ! video/x-raw,format=RGB ! "
                    f"tensor_converter ! "
                    f"tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! "
                    f"identity name=infer_start signal-handoffs=true ! "
                    # Blaze face detection model
                    f"tensor_filter framework=tensorflow2-lite "
                    f"model={self.model_blaze_path} custom=delegate:xnnpack ! "
                    # Blaze decode plugin
                    f"tensor_filter framework=blaze_decode model=dummy "
                    f"custom={self.width},{self.height} ! "
                    f"identity name=infer_end signal-handoffs=true ! "
                    f"crop.info "

                    f"crop. ! "
                    f"queue max-size-buffers=2 leaky=downstream ! "
                    f"crop_decode ! "

                    # Face landmark detection
                    f"tensor_filter framework=tensorflow2-lite model={self.model_landmark_path} custom=delegate:xnnpack ! "

                    # Decode + Ear detection
                    f"tensor_filter framework=face_mesh_decode model=dummy1 custom={self.width},{self.height} ! "
                    f"tensor_filter framework=fatigue_eval model=dummy3 ! "

                    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "  
                )
        else:
            # No use of model, just raw frames and debug          
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"videoconvert ! "
                # Split pipeline
                f"tee name=t "

                # Branch 1: Raw frames
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
                f"appsink name=appsink emit-signals=true max-buffers=1 drop=true "

                # Branch 2: Debug display
                f"t. ! queue max-size-buffers=2 leaky=downstream ! "
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
                
                # log.info("Inference result len of res_array: %s", len(res_array))
                # log.info("Global min: %f", res_array.min())
                # log.info("Global max: %f", res_array.max())

                # log.info("First 10 values: %s", res_array[:10])
                
                if len(res_array) == self.output_fatigue_dim and np.any(res_array):
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

    def on_new_crop_debug(self, sink, buffer):
        log.info("Got crop tensor buffer")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            log.error("Cannot map buffer")
            return

        try:
            raw = bytes(map_info.data)

            print("Buffer size:", len(raw))
            print("First 64 bytes (hex):", raw[:64].hex(" "))

            header_size = 128
            payload = raw[header_size:header_size + (self.output_crop_dim * 3)]

            img = np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width, 3)

            print("Image shape:", img.shape)
            print("Top-left pixel:", img[0, 0])
            print("Center pixel:", img[240, 320])
            print("Min/Max:", img.min(), img.max())

        except Exception as e:
            print("Error reading buffer:", e)

        finally:
            buffer.unmap(map_info)

    def on_new_bbox(self, sink, buffer):
        # log.info("Got bbox tensor buffer")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return

        try:
            # log.info("BBox buffer size (bytes): %d", map_info.size)
            raw = map_info.data

            header_size = 128
            payload = raw[header_size:header_size + 16]

            bbox = np.frombuffer(payload, dtype=np.float32)

            if len(bbox) == 4:
                x, y, w, h = bbox

                with self.draw_overlay_lock:
                    self.current_bbox = (x, y, w, h)

        except Exception as e:
            log.error("Error parsing bbox: %s", e)
        finally:
            buffer.unmap(map_info)

    def on_new_landmarks(self, sink, buffer):
        # log.info("Got landmarks tensor buffer")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return

        try:
            # log.info("Landmarks buffer size (bytes): %d", map_info.size)
            raw = map_info.data

            header_size = 0
            num_values = self.output_landmarks_dim

            payload_size = num_values * 4  # float32
            payload = raw[header_size:header_size + payload_size]
            landmarks = np.frombuffer(payload, dtype=np.float32)

            if landmarks.size == num_values:
                lm = landmarks.reshape(self.num_landmarks, self.landmarks_dim)

                with self.draw_overlay_lock:
                    self.current_landmarks = lm.copy()

            else:
                log.warning(
                    "Unexpected landmark size: got %d, expected %d",
                    landmarks.size, num_values
                )

        except Exception as e:
            log.error("Error parsing landmarks: %s", e)
        finally:
            buffer.unmap(map_info)

    def on_draw_overlay(self, overlay, context, timestamp, duration):
        with self.draw_overlay_lock:
            if self.current_landmarks is not None and self.current_bbox is not None and np.any(self.current_landmarks) and np.any(self.current_bbox):
                bx, by, bw, bh = self.current_bbox
                context.set_source_rgb(1, 0, 0)

                for lx, ly in self.current_landmarks:
                    nx = float(lx) / LANDMARK_INPUT_SIZE
                    ny = float(ly) / LANDMARK_INPUT_SIZE

                    x = bx + nx * bw
                    y = by + ny * bh

                    # log.info("Face mesh landmark: (%.2f, %.2f)", x, y)

                    context.arc(x, y, 3, 0, 2 * 3.1416)
                    context.fill()

        with self.fatigue_lock:
            data = self.current_fatigue_output.copy() if self.current_fatigue_output else None

        # draw info panel
        context.set_source_rgba(0, 0, 0, 0.5)
        context.rectangle(5, 5, 360, 290)
        context.fill()

        context.set_source_rgb(1, 1, 1)
        context.select_font_face("monospace", 0, 0)
        context.set_font_size(14)

        position_x = 15
        position_y = 25
        context.move_to(position_x, position_y)
        context.show_text(f"FPS: {self.pipeline_fps}")
        position_y += 20

        context.move_to(position_x, position_y)
        context.show_text(f"AI Latency: {self.avg_inference_time:.2f} ms")
        position_y += 20

        context.move_to(position_x, position_y)
        context.show_text(f"Pipeline Latency: {self.avg_pipeline_latency:.2f} ms")
        position_y += 20

        # ===== Fatigue Output =====
        if data is not None:
            state_names = {
                0: "NORMAL",
                1: "WARNING",
                2: "TIRED",
                3: "DISTRACTED",
                4: "NO_FACE",
            }

            context.move_to(position_x, position_y)
            context.show_text(f"State: {state_names.get(data['state'], 'UNK')}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Fatigue: {data['fatigue_score']:.2f}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Distraction: {data['distraction_score']:.2f}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"EAR L/R: {data['left_ear']:.3f} / {data['right_ear']:.3f}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Blink Rate: {data['blink_rate_per_min']:.1f}/min")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Eye Closed: {data['closed_duration_ms']:.0f} ms")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"MAR: {data['mar']:.3f}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Yawn: {data['yawn_hold_ms']:.0f} ms")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Head Roll: {data['head_roll_deg']:.1f}")
            position_y += 20

            context.move_to(position_x, position_y)
            context.show_text(f"Yaw/Pitch: {data['head_yaw_proxy']:.3f} / {data['head_pitch_proxy']:.3f}")
            position_y += 20

        else:
            context.move_to(position_x, position_y)
            context.show_text("Fatigue: No data")
    
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
                        # log.info("--- [AI DATA] ---")
                            # log.info(f"Number of data: {len(raw_data)}")
                            # log.info(f"First 5 data: {raw_data[:5]}")
                            # log.info("-" * 30)

                        fatigue_data = {
                            "state": int(raw_data[0]),
                            "fatigue_score": float(raw_data[1]),
                            "distraction_score": float(raw_data[2]),
                            "left_ear": float(raw_data[3]),
                            "right_ear": float(raw_data[4]),
                            "blink_rate_per_min": float(raw_data[5]),
                            "closed_duration_ms": float(raw_data[6]),
                            "mar": float(raw_data[7]),
                            "yawn_hold_ms": float(raw_data[8]),
                            "head_roll_deg": float(raw_data[9]),
                            "head_yaw_proxy": float(raw_data[10]),
                            "head_pitch_proxy": float(raw_data[11]),
                            "gaze_x_proxy": float(raw_data[12]),
                            "gaze_y_proxy": float(raw_data[13]),
                        }
                        
                        with self.fatigue_lock:
                            self.current_fatigue_output = fatigue_data
                            self.current_state = fatigue_data["state"]
                    else:
                        with self.fatigue_lock:
                            self.current_fatigue_output = None
                            self.current_state = F_STATE_NO_FACE
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

    model_blaze_path = MODEL_BLAZE_FACE_DETECTION
    model_landmark_path = MODEL_LANDMARK_FACE_DETECTION

    engine = ImageAnalyticsEngine(width=640, height=480, fps=args.fps,
                               model_blaze_path=model_blaze_path, model_landmark_path=model_landmark_path,
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