#!/usr/bin/env python3
import os
import glob
import csv
import time
import threading
import logging
import argparse

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dataset-logger")

Gst.init(None)

# ===== Models =====
MODEL_BLAZE_FACE_DETECTION = "/usr/share/models/blaze_face_short_range.tflite"
MODEL_LANDMARK_FACE_DETECTION = "/usr/share/models/face_landmark.tflite"

# ===== Constants =====
F_STATE_NORMAL = 0
F_STATE_WARNING = 1
F_STATE_TIRED = 2
F_STATE_DISTRACTED = 3
F_STATE_NO_FACE = 4

NUM_LANDMARKS = 468
LANDMARK_DIM = 2
OUTPUT_DIM = 14

EAR_CLOSE_THRESHOLD = 0.18
EAR_WARNING_THRESHOLD = 0.23
EAR_WARNING_HOLD_MS = 500
EAR_CLOSED_HOLD_MS = 1500

BLINK_LOW_THRESHOLD_PER_MIN = 8.0
MAR_YAWN_THRESHOLD = 0.35
YAWN_HOLD_MS = 800

HEAD_ROLL_THRESHOLD_DEG = 15.0
HEAD_YAW_PROXY_THRESHOLD = 0.18
HEAD_PITCH_PROXY_THRESHOLD = 0.20

LANDMARK_INPUT_SIZE = 192


class DatasetFeatureLogger:
    def __init__(
        self,
        dataset_root: str,
        csv_out: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        model_blaze_path: str = MODEL_BLAZE_FACE_DETECTION,
        model_landmark_path: str = MODEL_LANDMARK_FACE_DETECTION,
    ):
        self.dataset_root = dataset_root
        self.csv_out = csv_out
        self.width = width
        self.height = height
        self.fps = fps
        self.model_blaze_path = model_blaze_path
        self.model_landmark_path = model_landmark_path

        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.main_loop = GLib.MainLoop()

        self.current_label = 0
        self.current_class = ""
        self.current_file = ""

        self.csv_fh = None
        self.csv_writer = None

        self.feed_thread = None
        self.stop_event = threading.Event()

    def open_csv(self):
        new_file = not os.path.exists(self.csv_out)
        self.csv_fh = open(self.csv_out, "a", newline="")
        self.csv_writer = csv.writer(self.csv_fh)

        if new_file:
            self.csv_writer.writerow([
                "class",
                "label",
                "source_file",
                "state",
                "fatigue_score",
                "distraction_score",
                "left_ear",
                "right_ear",
                "blink_rate_per_min",
                "closed_duration_ms",
                "mar",
                "yawn_hold_ms",
                "head_roll_deg",
                "head_yaw_proxy",
                "head_pitch_proxy",
                "gaze_x_proxy",
                "gaze_y_proxy",
                "eye_closed_score",
                "blink_low_score",
                "yawn_score",
                "posture_score",
            ])
            self.csv_fh.flush()

    def close_csv(self):
        if self.csv_fh:
            self.csv_fh.flush()
            self.csv_fh.close()
            self.csv_fh = None
            self.csv_writer = None

    @staticmethod
    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    @staticmethod
    def reconstruct_scores(arr: np.ndarray):
        """
        Reconstruct the component scores from the raw outputs of fatigue_eval.
        Output layout:
        [0] state
        [1] fatigue_score
        [2] distraction_score
        [3] left_EAR
        [4] right_EAR
        [5] blink_rate_per_min
        [6] closed_duration_ms
        [7] MAR
        [8] yawn_hold_ms
        [9] head_roll_deg
        [10] head_yaw_proxy
        [11] head_pitch_proxy
        [12] gaze_x_proxy
        [13] gaze_y_proxy
        """
        closed_ms = float(arr[6])
        blink_rate = float(arr[5])
        mar = float(arr[7])
        yawn_hold_ms = float(arr[8])
        head_roll_deg = float(arr[9])
        head_yaw_proxy = float(arr[10])
        head_pitch_proxy = float(arr[11])
        gaze_x_proxy = float(arr[12])

        # Eye closed score
        if closed_ms >= EAR_CLOSED_HOLD_MS:
            eye_closed_score = 1.0
        elif closed_ms > EAR_WARNING_HOLD_MS:
            eye_closed_score = (closed_ms - EAR_WARNING_HOLD_MS) / (EAR_CLOSED_HOLD_MS - EAR_WARNING_HOLD_MS)
        else:
            eye_closed_score = 0.0

        # Blink low score
        if blink_rate <= BLINK_LOW_THRESHOLD_PER_MIN:
            blink_low_score = (BLINK_LOW_THRESHOLD_PER_MIN - blink_rate) / BLINK_LOW_THRESHOLD_PER_MIN
        else:
            blink_low_score = 0.0
        blink_low_score = DatasetFeatureLogger.clamp01(blink_low_score)

        # Yawn score
        if yawn_hold_ms >= YAWN_HOLD_MS:
            yawn_score = 1.0
        else:
            yawn_score = DatasetFeatureLogger.clamp01(mar / (MAR_YAWN_THRESHOLD * 1.5))

        # Posture score
        posture_score = (
            DatasetFeatureLogger.clamp01(abs(head_yaw_proxy) / HEAD_YAW_PROXY_THRESHOLD) * 0.4 +
            DatasetFeatureLogger.clamp01(abs(head_pitch_proxy) / HEAD_PITCH_PROXY_THRESHOLD) * 0.4 +
            DatasetFeatureLogger.clamp01(abs(head_roll_deg) / HEAD_ROLL_THRESHOLD_DEG) * 0.2
        )
        posture_score = DatasetFeatureLogger.clamp01(posture_score)

        return eye_closed_score, blink_low_score, yawn_score, posture_score

    def build_pipeline_str(self):
        """
        Pipeline source is appsrc (instead of libcamerasrc).
        The rest is kept close to your current pipeline.
        """
        pipeline = (
            f'appsrc name=src is-live=false block=true format=time '
            f'caps=video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1 ! '
            f'tee name=t '
            f'tensor_crop name=crop silent=false '
            f't. ! queue max-size-buffers=2 leaky=downstream ! '
            f'videoconvert ! video/x-raw,format=RGB ! '
            f'tensor_converter ! '
            f'crop.raw '
            f't. ! queue max-size-buffers=2 leaky=downstream ! '
            f'videoscale ! video/x-raw,width=128,height=128 ! '
            f'videoconvert ! video/x-raw,format=RGB ! '
            f'tensor_converter ! '
            f'tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! '
            f'identity name=infer_start signal-handoffs=true ! '
            f'tensor_filter framework=tensorflow2-lite '
            f'model={self.model_blaze_path} custom=delegate:xnnpack ! '
            f'tensor_filter framework=blaze_decode model=dummy '
            f'custom={self.width},{self.height} ! '
            f'identity name=infer_end signal-handoffs=true ! '
            f'crop.info '
            f'crop. ! '
            f'queue max-size-buffers=2 leaky=downstream ! '
            f'crop_decode ! '
            f'tensor_filter framework=tensorflow2-lite '
            f'model={self.model_landmark_path} custom=delegate:xnnpack ! '
            f'tensor_filter framework=face_mesh_decode model=dummy1 custom={self.width},{self.height} ! '
            f'tensor_filter framework=fatigue_eval model=dummy3 ! '
            f'appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true'
        )
        return pipeline

    def on_bus_message(self, bus, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("GStreamer error: %s", err.message)
            log.error("Debug: %s", debug)
            self.stop_event.set()
            self.main_loop.quit()

        elif msg_type == Gst.MessageType.EOS:
            log.info("EOS received")
            self.stop_event.set()
            self.main_loop.quit()

    def on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        try:
            arr = np.frombuffer(map_info.data, dtype=np.float32).copy()

            if arr.size != OUTPUT_DIM:
                return Gst.FlowReturn.OK

            state = int(arr[0])

            # Skip no-face frames if you want a cleaner training set
            if state == F_STATE_NO_FACE:
                return Gst.FlowReturn.OK

            eye_closed_score, blink_low_score, yawn_score, posture_score = self.reconstruct_scores(arr)

            self.csv_writer.writerow([
                self.current_class,
                self.current_label,
                self.current_file,
                state,
                float(arr[1]),   # fatigue_score
                float(arr[2]),   # distraction_score
                float(arr[3]),   # left_EAR
                float(arr[4]),   # right_EAR
                float(arr[5]),   # blink_rate_per_min
                float(arr[6]),   # closed_duration_ms
                float(arr[7]),   # MAR
                float(arr[8]),   # yawn_hold_ms
                float(arr[9]),   # head_roll_deg
                float(arr[10]),  # head_yaw_proxy
                float(arr[11]),  # head_pitch_proxy
                float(arr[12]),  # gaze_x_proxy
                float(arr[13]),  # gaze_y_proxy
                float(eye_closed_score),
                float(blink_low_score),
                float(yawn_score),
                float(posture_score),
            ])
            self.csv_fh.flush()

        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _feed_images(self, folder_path: str):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder_path, p)))
        files = sorted(files)

        if not files:
            log.warning("No images found in %s", folder_path)
            try:
                self.appsrc.emit("end-of-stream")
            except Exception:
                pass
            return

        duration = int(Gst.SECOND / self.fps)

        for idx, file_path in enumerate(files):
            if self.stop_event.is_set():
                break

            img = cv2.imread(file_path)
            if img is None:
                log.warning("Cannot read image: %s", file_path)
                continue

            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            data = img.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            pts = idx * duration
            buf.pts = pts
            buf.dts = pts
            buf.duration = duration

            self.current_file = os.path.basename(file_path)

            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                log.warning("push-buffer returned %s at %s", ret, file_path)
                break

        try:
            self.appsrc.emit("end-of-stream")
        except Exception as e:
            log.warning("Failed to send EOS: %s", e)

    def run_folder(self, folder_path: str, label: int):
        self.current_label = label
        self.current_class = os.path.basename(os.path.normpath(folder_path))

        self.stop_event.clear()
        self.pipeline = Gst.parse_launch(self.build_pipeline_str())

        self.appsrc = self.pipeline.get_by_name("src")
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsrc is None:
            raise RuntimeError("appsrc element not found")
        if self.appsink is None:
            raise RuntimeError("appsink element not found")

        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.connect("new-sample", self.on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        log.info("Starting folder: %s (label=%d)", folder_path, label)
        self.pipeline.set_state(Gst.State.PLAYING)

        self.feed_thread = threading.Thread(
            target=self._feed_images,
            args=(folder_path,),
            daemon=True,
        )
        self.feed_thread.start()

        try:
            self.main_loop.run()
        except Exception as e:
            log.exception("Main loop error: %s", e)
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            if self.feed_thread.is_alive():
                self.feed_thread.join(timeout=2.0)

        log.info("Finished folder: %s", folder_path)

    def run(self):
        self.open_csv()
        try:
            class_map = [
                ("normal", 0),
                ("tired", 1),
            ]

            for class_name, label in class_map:
                folder = os.path.join(self.dataset_root, class_name)
                if not os.path.isdir(folder):
                    log.warning("Folder not found: %s", folder)
                    continue
                self.run_folder(folder, label)

        finally:
            self.close_csv()


def main():
    parser = argparse.ArgumentParser(description="Dataset pipeline logger")
    parser.add_argument("--dataset-root", required=True, help="Root folder containing normal/ and tired/")
    parser.add_argument("--csv-out", default="data.csv", help="Output CSV file")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--blaze-model", default=MODEL_BLAZE_FACE_DETECTION)
    parser.add_argument("--landmark-model", default=MODEL_LANDMARK_FACE_DETECTION)
    args = parser.parse_args()

    engine = DatasetFeatureLogger(
        dataset_root=args.dataset_root,
        csv_out=args.csv_out,
        width=args.width,
        height=args.height,
        fps=args.fps,
        model_blaze_path=args.blaze_model,
        model_landmark_path=args.landmark_model,
    )
    engine.run()


if __name__ == "__main__":
    main()