import argparse
from collections import deque
import logging
import signal
import struct
import sys

import gi
import cairo

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GstPersonDetector")


class GstPersonDetector:
	"""
	GStreamer-based real-time person detection system using SSD MobileNet V3 (TFLite).

	Features:
	- Capture video from camera via libcamera (through GStreamer pipeline)
	- Run inference on resized frames at a configurable FPS
	- Detect persons and draw real-time bounding boxes
	- Apply post-processing (NMS, duplicate merging) for stable detections
	- Smooth detection count over a temporal window to reduce noise
	- Optional live preview via configurable video sink (kmssink, waylandsink, ...)

	Pipeline:
	1. Capture frames from camera
	2. Resize to model input size (320x320)
	3. Convert to RGB
	4. Run TFLite inference with XNNPACK delegate
	5. Post-process detections (filter, NMS, merge)
	6. Smooth person count over time
	7. Overlay results and display output
	"""

	def __init__(
		self,
		model_path: str,
		camera_width: int = 640,
		camera_height: int = 480,
		model_size: int = 320,
		threshold: float = 0.5,
		nms_iou_threshold: float = 0.3,
		min_box_area_ratio: float = 0.01,
		merge_center_dist: float = 0.06,
		smooth_window: int = 3,
		ai_fps: int = 8,
		ai_threads: int = 4,
		display_sink: str = "none",
	):
		self.model_path = model_path
		self.camera_width = camera_width
		self.camera_height = camera_height
		self.model_size = model_size
		self.threshold = threshold
		self.nms_iou_threshold = max(0.0, min(1.0, nms_iou_threshold))
		self.min_box_area_ratio = max(0.0, min(1.0, min_box_area_ratio))
		self.merge_center_dist = max(0.0, min(1.0, merge_center_dist))
		self.smooth_window = max(1, smooth_window)
		self.ai_fps = ai_fps
		self.ai_threads = ai_threads
		self.display_sink = (display_sink or "none").strip().lower()

		self.pipeline = None
		self.loop = None
		self.sample_count = 0
		self.overlay_boxes: list[tuple[float, float, float, float]] = []
		self.last_person_count = -1
		self.count_history: deque[int] = deque(maxlen=self.smooth_window)

		Gst.init(None)
		self._build_pipeline()
		self._setup_signals()

	def _build_pipeline(self) -> None:
		"""
		Build GStreamer pipeline for camera capture and inference.
		
		When display sink is set to 'none':
		- Runs AI branch only (no preview output)

		When display sink is set (e.g. kmssink, waylandsink):
		- AI branch runs inference
		- Display branch shows preview with overlayed detections
		
		Model input: uint8 tensor [1,320,320,3] in RGB format
		"""
		ai_branch = (
			f"queue leaky=downstream max-size-buffers=1 ! "
			f"videorate ! video/x-raw,framerate={self.ai_fps}/1 ! "
			f"videoscale ! video/x-raw,width={self.model_size},height={self.model_size} ! "
			f"videoconvert ! video/x-raw,format=RGB ! "
			f"tensor_converter ! "
			f"tensor_filter framework=tensorflow-lite model={self.model_path} "
			f"custom=Delegate:XNNPACK,NumThreads:{self.ai_threads} ! "
			f"tensor_sink name=det_sink"
		)

		if self.display_sink == "none":
			pipeline_str = (
				f"libcamerasrc ! "
				f"video/x-raw,width={self.camera_width},height={self.camera_height} ! "
				f"{ai_branch}"
			)
		else:
			pipeline_str = (
				f"libcamerasrc ! "
				f"video/x-raw,width={self.camera_width},height={self.camera_height} ! "
				f"tee name=t "

				f"t. ! {ai_branch} "

				f"t. ! queue leaky=downstream max-size-buffers=5 ! "
				f"videoconvert ! cairooverlay name=person_overlay ! "
				f"videoconvert ! video/x-raw,format=I420,width={self.camera_width},height={self.camera_height} ! "
				f"{self.display_sink}"
			)

		try:
			logger.info("Initializing pipeline...")
			self.pipeline = Gst.parse_launch(pipeline_str)
		except GLib.Error as e:
			logger.error("Failed to build pipeline: %s", e)
			sys.exit(1)

		bus = self.pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self._on_bus_message)

		det_sink = self.pipeline.get_by_name("det_sink")
		if not det_sink:
			logger.error("tensor_sink 'det_sink' not found")
			sys.exit(1)
		det_sink.connect("new-data", self._on_new_data)

		if self.display_sink != "none":
			overlay = self.pipeline.get_by_name("person_overlay")
			if not overlay:
				logger.error("cairooverlay 'person_overlay' not found")
				sys.exit(1)
			overlay.connect("draw", self._on_overlay_draw)

	def _setup_signals(self) -> None:
		signal.signal(signal.SIGINT, self._on_interrupt)
		signal.signal(signal.SIGTERM, self._on_interrupt)

	def _on_interrupt(self, sig: int, _frame) -> None:
		logger.warning("Caught signal %s. Shutting down.", sig)
		self.shutdown()

	def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
		msg_type = message.type
		if msg_type == Gst.MessageType.ERROR:
			err, dbg = message.parse_error()
			logger.error("GStreamer error: %s | %s", err, dbg)
			self.shutdown()
		elif msg_type == Gst.MessageType.EOS:
			logger.info("EOS received.")
			self.shutdown()

	def _on_new_data(self, _sink: Gst.Element, buffer: Gst.Buffer) -> None:
		self.sample_count += 1
		parsed = self._parse_postprocess_fast(buffer)
		if parsed is None:
			if self.sample_count % 30 == 0:
				logger.info("Waiting valid postprocess tensors...")
			return

		person_count, boxes = parsed
		self.count_history.append(person_count)
		smoothed_count = int(round(sum(self.count_history) / len(self.count_history)))

		self.overlay_boxes = boxes

		if smoothed_count != self.last_person_count:
			print(f"[PERSON] count={smoothed_count}", flush=True)
			self.last_person_count = smoothed_count

	def _on_overlay_draw(self, _overlay, context, _timestamp, _duration) -> None:
		if not self.overlay_boxes:
			return

		context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		context.set_font_size(18.0)

		for x1, y1, x2, y2 in self.overlay_boxes:
			# Draw bounding box
			context.set_source_rgb(0.0, 1.0, 0.0)  # Green box
			context.set_line_width(2.0)
			context.rectangle(x1, y1, x2 - x1, y2 - y1)
			context.stroke()

			# Draw label background
			context.set_source_rgba(0.0, 1.0, 0.0, 0.8)  # Green with transparency
			extents = context.text_extents("person")
			pad = 3
			label_width = extents.width + pad * 2
			label_height = extents.height + pad * 2
			context.rectangle(x1, y1 - label_height - 2, label_width, label_height)
			context.fill()

			# Draw label text
			context.move_to(x1 + pad, y1 - 4)
			context.set_source_rgb(0.0, 0.0, 0.0)  # Black text
			context.show_text("person")

	def _parse_postprocess_fast(self, buffer: Gst.Buffer) -> tuple[int, list[tuple[float, float, float, float]]] | None:
		"""
		Parse and post-process TFLite detection outputs.
		
		Tensor layout (from TFLite_Detection_PostProcess):
		- mem0: Bounding boxes [N,4] - (ymin, xmin, ymax, xmax) in normalized coords
		- mem1: Class IDs [N] - filter for class 0 (person only)
		- mem2: Confidence scores [N] - filter by threshold
		- mem3: Detection count [1]
		
		Post-processing steps:
		1. Filter by confidence threshold
		2. Filter by class ID (0 = person)
		3. Filter by minimum box area
		4. Apply NMS (IoU-based deduplication)
		5. Merge near-duplicate boxes by center distance
		6. Convert to camera coordinates for display
		"""
		if buffer.n_memory() < 4:
			return None

		def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
			ax1, ay1, ax2, ay2 = a
			bx1, by1, bx2, by2 = b
			inter_x1 = max(ax1, bx1)
			inter_y1 = max(ay1, by1)
			inter_x2 = min(ax2, bx2)
			inter_y2 = min(ay2, by2)
			inter_w = max(0.0, inter_x2 - inter_x1)
			inter_h = max(0.0, inter_y2 - inter_y1)
			inter_area = inter_w * inter_h
			if inter_area <= 0.0:
				return 0.0
			area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
			area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
			union = area_a + area_b - inter_area
			if union <= 0.0:
				return 0.0
			return inter_area / union

		def _map_floats(mem_idx: int):
			mem = buffer.peek_memory(mem_idx)
			ok, info = mem.map(Gst.MapFlags.READ)
			if not ok:
				return None, None, None
			data = memoryview(info.data)
			if len(data) < 4 or len(data) % 4 != 0:
				mem.unmap(info)
				return None, None, None
			return mem, info, data

		mem3, info3, data3 = _map_floats(3)
		if not data3:
			return None
		try:
			det_count = int(struct.unpack_from("<f", data3, 0)[0])
		finally:
			mem3.unmap(info3)

		if det_count <= 0:
			return 0, []
		det_count = min(det_count, 50)

		mem1, info1, data1 = _map_floats(1)
		mem2, info2, data2 = _map_floats(2)
		mem0, info0, data0 = _map_floats(0)
		if not data1 or not data2 or not data0:
			if data1:
				mem1.unmap(info1)
			if data2:
				mem2.unmap(info2)
			if data0:
				mem0.unmap(info0)
			return None

		try:
			max_classes = len(data1) // 4
			max_scores = len(data2) // 4
			max_boxes = (len(data0) // 4) // 4
			n = min(det_count, max_classes, max_scores, max_boxes)
			if n <= 0:
				return 0, []

			# Keep candidates first, then apply NMS to reduce duplicate detections
			# for the same person in a single frame.
			candidates: list[tuple[float, float, float, float, float]] = []
			for i in range(n):
				score = struct.unpack_from("<f", data2, i * 4)[0]
				if score < self.threshold:
					continue

				class_id = int(struct.unpack_from("<f", data1, i * 4)[0])
				if class_id != 0:
					continue

				box_base = i * 16
				ymin = struct.unpack_from("<f", data0, box_base)[0]
				xmin = struct.unpack_from("<f", data0, box_base + 4)[0]
				ymax = struct.unpack_from("<f", data0, box_base + 8)[0]
				xmax = struct.unpack_from("<f", data0, box_base + 12)[0]

				xmin = max(0.0, min(1.0, xmin))
				ymin = max(0.0, min(1.0, ymin))
				xmax = max(0.0, min(1.0, xmax))
				ymax = max(0.0, min(1.0, ymax))
				if xmax <= xmin or ymax <= ymin:
					continue
				box_area = (xmax - xmin) * (ymax - ymin)
				if box_area < self.min_box_area_ratio:
					continue

				candidates.append((score, xmin, ymin, xmax, ymax))

			if not candidates:
				return 0, []

			# SSD postprocess usually outputs sorted by score, but sort anyway for safety.
			candidates.sort(key=lambda item: item[0], reverse=True)

			selected: list[tuple[float, float, float, float, float]] = []
			for _score, xmin, ymin, xmax, ymax in candidates:
				box = (xmin, ymin, xmax, ymax)
				if any(_iou(box, (kx1, ky1, kx2, ky2)) >= self.nms_iou_threshold for _, kx1, ky1, kx2, ky2 in selected):
					continue
				selected.append((_score, xmin, ymin, xmax, ymax))

			# Merge residual near-duplicate boxes that may survive IoU-based NMS
			# when one box is nested/offset and IoU is low.
			merged: list[tuple[float, float, float, float, float]] = []
			for score, xmin, ymin, xmax, ymax in selected:
				cx = (xmin + xmax) * 0.5
				cy = (ymin + ymax) * 0.5
				keep = True
				for _, mx1, my1, mx2, my2 in merged:
					mcx = (mx1 + mx2) * 0.5
					mcy = (my1 + my2) * 0.5
					dx = cx - mcx
					dy = cy - mcy
					if (dx * dx + dy * dy) ** 0.5 <= self.merge_center_dist:
						keep = False
						break
				if keep:
					merged.append((score, xmin, ymin, xmax, ymax))

			boxes: list[tuple[float, float, float, float]] = []
			for _score, xmin, ymin, xmax, ymax in merged:
				x1 = max(0.0, min(float(self.camera_width - 1), xmin * self.camera_width))
				y1 = max(0.0, min(float(self.camera_height - 1), ymin * self.camera_height))
				x2 = max(0.0, min(float(self.camera_width - 1), xmax * self.camera_width))
				y2 = max(0.0, min(float(self.camera_height - 1), ymax * self.camera_height))
				boxes.append((x1, y1, x2, y2))

			return len(merged), boxes
		finally:
			mem1.unmap(info1)
			mem2.unmap(info2)
			mem0.unmap(info0)

	def start(self) -> None:
		if not self.pipeline:
			return
		if self.display_sink == "none":
			logger.info("Starting person counting mode (headless, no preview sink)...")
		else:
			logger.info("Starting person counting mode with preview sink: %s", self.display_sink)
		self.pipeline.set_state(Gst.State.PLAYING)
		self.loop = GLib.MainLoop()
		try:
			self.loop.run()
		except Exception as e:
			logger.error("Main loop failed: %s", e)
			self.shutdown()

	def shutdown(self) -> None:
		if self.pipeline:
			self.pipeline.set_state(Gst.State.NULL)
		if self.loop and self.loop.is_running():
			self.loop.quit()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Print-only SSD MobileNet V3 TFLite detector (postprocess outputs)."
	)
	parser.add_argument("--model", required=True, help="Path to SSD MobileNet V3 .tflite model")
	parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
	parser.add_argument("--nms-iou", type=float, default=0.2, help="IoU threshold for NMS")
	parser.add_argument("--min-box-area", type=float, default=0.01, help="Min normalized box area")
	parser.add_argument("--merge-center-dist", type=float, default=0.06, help="Center distance to merge duplicates")
	parser.add_argument("--smooth-window", type=int, default=5, help="Frame window for count smoothing")
	parser.add_argument("--ai-fps", type=int, default=6, help="Inference FPS for AI branch")
	parser.add_argument("--ai-threads", type=int, default=4, help="XNNPACK threads")
	parser.add_argument(
		"--display-sink",
		type=str,
		default="none",
		help="Display sink to enable preview (none, kmssink, waylandsink, ...)",
	)
	parser.add_argument("--camera-width", type=int, default=640)
	parser.add_argument("--camera-height", type=int, default=480)
	parser.add_argument("--model-size", type=int, default=320)
	args = parser.parse_args()

	launcher = GstPersonDetector(
		model_path=args.model,
		camera_width=args.camera_width,
		camera_height=args.camera_height,
		model_size=args.model_size,
		threshold=args.threshold,
		nms_iou_threshold=args.nms_iou,
		min_box_area_ratio=args.min_box_area,
		merge_center_dist=args.merge_center_dist,
		smooth_window=args.smooth_window,
		ai_fps=args.ai_fps,
		ai_threads=args.ai_threads,
		display_sink=args.display_sink,
	)
	launcher.start()


if __name__ == "__main__":
	main()
