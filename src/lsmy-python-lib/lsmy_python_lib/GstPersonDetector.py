"""
Optimized GStreamer person detector using SSD MobileNet V3 (TFLite).

Design notes:
- Camera FPS is constrained at the source caps to cap capture bandwidth.
- AI branch uses videorate drop-only=true to never duplicate frames.
- Post-processing is pure Python (struct.unpack_from) to keep CPU usage low on
  Raspberry Pi where numpy import + per-frame array allocations dominate.
- Inference timing is tracked via a bounded OrderedDict (no leak on dropped frames)
  and reported as an EMA throttled to once per second to avoid log I/O overhead.
- overlay_boxes is published as an immutable tuple so the cairooverlay reader
  thread always sees a consistent snapshot under the GIL.
- Display branch keeps the same conversion graph as upstream — eliminating the
  second videoconvert is sink-specific and was deliberately left out.
"""
from __future__ import annotations

import argparse
import logging
import signal
import struct
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Optional

import cairo
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst  # noqa: E402

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PersonDetector")


PERSON_CLASS_ID = 0
MAX_DETECTIONS = 50
INFER_TS_CAPACITY = 32
LOG_INTERVAL_S = 1.0


@dataclass
class DetectorConfig:
	model_path: str
	camera_width: int = 640
	camera_height: int = 480
	camera_fps: int = 30  # 0 disables source-side rate cap
	model_size: int = 320
	threshold: float = 0.5
	nms_iou_threshold: float = 0.3
	min_box_area_ratio: float = 0.01
	merge_center_dist: float = 0.06
	smooth_window: int = 3
	ai_fps: int = 8
	ai_threads: int = 4
	display_sink: str = "none"

	def __post_init__(self) -> None:
		self.threshold = _clip01(self.threshold)
		self.nms_iou_threshold = _clip01(self.nms_iou_threshold)
		self.min_box_area_ratio = _clip01(self.min_box_area_ratio)
		self.merge_center_dist = _clip01(self.merge_center_dist)
		self.smooth_window = max(1, self.smooth_window)
		self.display_sink = (self.display_sink or "none").strip().lower()


def _clip01(x: float) -> float:
	return max(0.0, min(1.0, x))


class PostProcessor:
	"""
	Pure-Python post-processing for TFLite_Detection_PostProcess output tensors.

	Tensor layout:
	  mem0: boxes [N,4]  (ymin, xmin, ymax, xmax) normalized
	  mem1: classes [N]
	  mem2: scores [N]
	  mem3: count [1]
	"""

	def __init__(self, cfg: DetectorConfig):
		self.threshold = cfg.threshold
		self.iou_thr = cfg.nms_iou_threshold
		self.min_area = cfg.min_box_area_ratio
		self.merge_dist_sq = cfg.merge_center_dist ** 2
		self.cam_w = cfg.camera_width
		self.cam_h = cfg.camera_height
		self._w_max = float(cfg.camera_width - 1)
		self._h_max = float(cfg.camera_height - 1)

	@staticmethod
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

	def parse(self, buffer: Gst.Buffer) -> Optional[tuple[int, list[tuple[float, float, float, float]]]]:
		if buffer.n_memory() < 4:
			return None

		# Read detection count first; bail out cheaply when there are no detections
		# without paying for the box/class/score memory maps.
		mem3 = buffer.peek_memory(3)
		ok3, info3 = mem3.map(Gst.MapFlags.READ)
		if not ok3:
			return None
		try:
			data3 = info3.data
			if len(data3) < 4 or len(data3) % 4 != 0:
				return None
			det_count = int(struct.unpack_from("<f", data3, 0)[0])
		finally:
			mem3.unmap(info3)

		if det_count <= 0:
			return 0, []
		det_count = min(det_count, MAX_DETECTIONS)

		mapped: list[tuple[Gst.Memory, Gst.MapInfo]] = []
		try:
			buffers: list[memoryview] = []
			for idx in (1, 2, 0):
				mem = buffer.peek_memory(idx)
				ok, info = mem.map(Gst.MapFlags.READ)
				if not ok:
					return None
				mapped.append((mem, info))
				data = info.data
				if len(data) < 4 or len(data) % 4 != 0:
					return None
				buffers.append(memoryview(data))

			data1, data2, data0 = buffers

			max_classes = len(data1) // 4
			max_scores = len(data2) // 4
			max_boxes = (len(data0) // 4) // 4
			n = min(det_count, max_classes, max_scores, max_boxes)
			if n <= 0:
				return 0, []

			threshold = self.threshold
			min_area = self.min_area
			candidates: list[tuple[float, float, float, float, float]] = []
			for i in range(n):
				score = struct.unpack_from("<f", data2, i * 4)[0]
				if score < threshold:
					continue

				class_id = int(struct.unpack_from("<f", data1, i * 4)[0])
				if class_id != PERSON_CLASS_ID:
					continue

				box_base = i * 16
				ymin = struct.unpack_from("<f", data0, box_base)[0]
				xmin = struct.unpack_from("<f", data0, box_base + 4)[0]
				ymax = struct.unpack_from("<f", data0, box_base + 8)[0]
				xmax = struct.unpack_from("<f", data0, box_base + 12)[0]

				if xmin < 0.0:
					xmin = 0.0
				elif xmin > 1.0:
					xmin = 1.0
				if ymin < 0.0:
					ymin = 0.0
				elif ymin > 1.0:
					ymin = 1.0
				if xmax < 0.0:
					xmax = 0.0
				elif xmax > 1.0:
					xmax = 1.0
				if ymax < 0.0:
					ymax = 0.0
				elif ymax > 1.0:
					ymax = 1.0

				if xmax <= xmin or ymax <= ymin:
					continue
				if (xmax - xmin) * (ymax - ymin) < min_area:
					continue

				candidates.append((score, xmin, ymin, xmax, ymax))
		finally:
			for mem, info in mapped:
				mem.unmap(info)

		if not candidates:
			return 0, []

		# SSD postprocess usually outputs sorted by score, but sort anyway for safety.
		candidates.sort(key=lambda item: item[0], reverse=True)

		iou_thr = self.iou_thr
		iou = self._iou
		selected: list[tuple[float, float, float, float, float]] = []
		for score, xmin, ymin, xmax, ymax in candidates:
			box = (xmin, ymin, xmax, ymax)
			suppressed = False
			for _, kx1, ky1, kx2, ky2 in selected:
				if iou(box, (kx1, ky1, kx2, ky2)) >= iou_thr:
					suppressed = True
					break
			if suppressed:
				continue
			selected.append((score, xmin, ymin, xmax, ymax))

		# Merge residual near-duplicate boxes that survive IoU-based NMS when one
		# box is nested/offset and IoU is low. Compare squared distance to skip sqrt.
		merge_dist_sq = self.merge_dist_sq
		merged: list[tuple[float, float, float, float, float]] = []
		for score, xmin, ymin, xmax, ymax in selected:
			cx = (xmin + xmax) * 0.5
			cy = (ymin + ymax) * 0.5
			keep = True
			for _, mx1, my1, mx2, my2 in merged:
				dx = cx - (mx1 + mx2) * 0.5
				dy = cy - (my1 + my2) * 0.5
				if dx * dx + dy * dy <= merge_dist_sq:
					keep = False
					break
			if keep:
				merged.append((score, xmin, ymin, xmax, ymax))

		cam_w = self.cam_w
		cam_h = self.cam_h
		w_max = self._w_max
		h_max = self._h_max
		boxes_out: list[tuple[float, float, float, float]] = []
		for _score, xmin, ymin, xmax, ymax in merged:
			x1 = xmin * cam_w
			if x1 < 0.0:
				x1 = 0.0
			elif x1 > w_max:
				x1 = w_max
			y1 = ymin * cam_h
			if y1 < 0.0:
				y1 = 0.0
			elif y1 > h_max:
				y1 = h_max
			x2 = xmax * cam_w
			if x2 < 0.0:
				x2 = 0.0
			elif x2 > w_max:
				x2 = w_max
			y2 = ymax * cam_h
			if y2 < 0.0:
				y2 = 0.0
			elif y2 > h_max:
				y2 = h_max
			boxes_out.append((x1, y1, x2, y2))

		return len(boxes_out), boxes_out


class GstPersonDetector:
	def __init__(self, cfg: DetectorConfig):
		self.cfg = cfg
		self.post = PostProcessor(cfg)

		self.pipeline: Optional[Gst.Pipeline] = None
		self.loop: Optional[GLib.MainLoop] = None
		self.sample_count = 0

		# Published to overlay reader thread as an immutable tuple snapshot.
		self.overlay_boxes: tuple[tuple[float, float, float, float], ...] = ()

		self.last_person_count = -1
		self.count_history: deque[int] = deque(maxlen=cfg.smooth_window)

		self._infer_start_ts: OrderedDict[int, float] = OrderedDict()
		self._infer_ema_ms = 0.0
		self._infer_log_ts = 0.0

		Gst.init(None)
		self._build_pipeline()
		signal.signal(signal.SIGINT, self._on_interrupt)
		signal.signal(signal.SIGTERM, self._on_interrupt)

	def _build_pipeline(self) -> None:
		cfg = self.cfg
		fps_caps = f",framerate={cfg.camera_fps}/1" if cfg.camera_fps > 0 else ""
		src_caps = f"video/x-raw,width={cfg.camera_width},height={cfg.camera_height}{fps_caps}"

		ai_branch = (
			f"queue leaky=downstream max-size-buffers=1 ! "
			f"videorate drop-only=true ! video/x-raw,framerate={cfg.ai_fps}/1 ! "
			f"videoscale ! video/x-raw,width={cfg.model_size},height={cfg.model_size} ! "
			f"videoconvert ! video/x-raw,format=RGB ! "
			f"tensor_converter ! "
			f"identity name=infer_start signal-handoffs=true ! "
			f"tensor_filter framework=tensorflow-lite model={cfg.model_path} "
			f"custom=Delegate:XNNPACK,NumThreads:{cfg.ai_threads} ! "
			f"identity name=infer_end signal-handoffs=true ! "
			f"tensor_sink name=det_sink"
		)

		if cfg.display_sink == "none":
			pipeline_str = f"libcamerasrc ! {src_caps} ! {ai_branch}"
		else:
			pipeline_str = (
				f"libcamerasrc ! {src_caps} ! tee name=t allow-not-linked=true "
				f"t. ! {ai_branch} "
				f"t. ! queue leaky=downstream max-size-buffers=5 ! "
				f"videoconvert ! cairooverlay name=person_overlay ! "
				f"videoconvert ! video/x-raw,format=I420,width={cfg.camera_width},height={cfg.camera_height} ! "
				f"{cfg.display_sink} sync=false"
			)

		logger.info("Initializing pipeline...")
		try:
			self.pipeline = Gst.parse_launch(pipeline_str)
		except GLib.Error as e:
			logger.error("Failed to build pipeline: %s", e)
			sys.exit(1)

		bus = self.pipeline.get_bus()
		bus.add_signal_watch()
		bus.connect("message", self._on_bus_message)

		self._connect("det_sink", "new-data", self._on_new_data)
		self._connect("infer_start", "handoff", self._on_infer_start)
		self._connect("infer_end", "handoff", self._on_infer_end)
		if cfg.display_sink != "none":
			self._connect("person_overlay", "draw", self._on_overlay_draw)

	def _connect(self, name: str, signal_name: str, handler) -> None:
		elem = self.pipeline.get_by_name(name)
		if not elem:
			logger.error("Pipeline element '%s' not found", name)
			sys.exit(1)
		elem.connect(signal_name, handler)

	def _on_interrupt(self, sig: int, _frame) -> None:
		logger.warning("Caught signal %s. Shutting down.", sig)
		self.shutdown()

	def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
		t = message.type
		if t == Gst.MessageType.ERROR:
			err, dbg = message.parse_error()
			logger.error("GStreamer error: %s | %s", err, dbg)
			self.shutdown()
		elif t == Gst.MessageType.EOS:
			logger.info("EOS received.")
			self.shutdown()

	def _on_infer_start(self, _elem: Gst.Element, buffer: Gst.Buffer) -> None:
		pts = buffer.pts
		if pts == Gst.CLOCK_TIME_NONE:
			return
		self._infer_start_ts[pts] = time.perf_counter()
		while len(self._infer_start_ts) > INFER_TS_CAPACITY:
			self._infer_start_ts.popitem(last=False)

	def _on_infer_end(self, _elem: Gst.Element, buffer: Gst.Buffer) -> None:
		pts = buffer.pts
		if pts == Gst.CLOCK_TIME_NONE:
			return
		t0 = self._infer_start_ts.pop(pts, None)
		if t0 is None:
			return
		dt_ms = (time.perf_counter() - t0) * 1000.0
		self._infer_ema_ms = (
			0.9 * self._infer_ema_ms + 0.1 * dt_ms if self._infer_ema_ms else dt_ms
		)
		now = time.perf_counter()
		if now - self._infer_log_ts >= LOG_INTERVAL_S:
			logger.info("inference EMA: %.2f ms", self._infer_ema_ms)
			self._infer_log_ts = now

	def _on_new_data(self, _sink: Gst.Element, buffer: Gst.Buffer) -> None:
		self.sample_count += 1
		parsed = self.post.parse(buffer)
		if parsed is None:
			if self.sample_count % 30 == 0:
				logger.info("Waiting for valid postprocess tensors...")
			return

		count, boxes = parsed
		self.count_history.append(count)
		smoothed = int(round(sum(self.count_history) / len(self.count_history)))

		self.overlay_boxes = tuple(boxes)

		if smoothed != self.last_person_count:
			print(f"[PERSON] count={smoothed}", flush=True)
			self.last_person_count = smoothed

	def _on_overlay_draw(self, _overlay, ctx, _ts, _dur) -> None:
		snapshot = self.overlay_boxes
		if not snapshot:
			return

		ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		ctx.set_font_size(18.0)
		extents = ctx.text_extents("person")
		pad = 3
		label_w = extents.width + pad * 2
		label_h = extents.height + pad * 2

		for x1, y1, x2, y2 in snapshot:
			ctx.set_source_rgb(0.0, 1.0, 0.0)
			ctx.set_line_width(2.0)
			ctx.rectangle(x1, y1, x2 - x1, y2 - y1)
			ctx.stroke()

			ctx.set_source_rgba(0.0, 1.0, 0.0, 0.8)
			ctx.rectangle(x1, y1 - label_h - 2, label_w, label_h)
			ctx.fill()

			ctx.move_to(x1 + pad, y1 - 4)
			ctx.set_source_rgb(0.0, 0.0, 0.0)
			ctx.show_text("person")

	def start(self) -> None:
		if not self.pipeline:
			return
		mode = "headless" if self.cfg.display_sink == "none" else f"preview ({self.cfg.display_sink})"
		logger.info("Starting person counting mode: %s", mode)
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


def _parse_args() -> DetectorConfig:
	p = argparse.ArgumentParser(description="Optimized SSD MobileNet V3 person detector.")
	p.add_argument("--model", required=True, help="Path to .tflite model")
	p.add_argument("--threshold", type=float, default=0.5)
	p.add_argument("--nms-iou", type=float, default=0.3)
	p.add_argument("--min-box-area", type=float, default=0.01)
	p.add_argument("--merge-center-dist", type=float, default=0.06)
	p.add_argument("--smooth-window", type=int, default=3)
	p.add_argument("--ai-fps", type=int, default=8)
	p.add_argument("--ai-threads", type=int, default=4)
	p.add_argument("--camera-fps", type=int, default=30, help="0 disables source rate cap")
	p.add_argument("--display-sink", type=str, default="none")
	p.add_argument("--camera-width", type=int, default=640)
	p.add_argument("--camera-height", type=int, default=480)
	p.add_argument("--model-size", type=int, default=320)
	args = p.parse_args()
	return DetectorConfig(
		model_path=args.model,
		camera_width=args.camera_width,
		camera_height=args.camera_height,
		camera_fps=args.camera_fps,
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


def main() -> None:
	GstPersonDetector(_parse_args()).start()


if __name__ == "__main__":
	main()
