"""
AI Virtual Whiteboard
=====================
Uses MediaPipe for hand tracking and OpenCV/Pillow for clean drawing and UI.

Controls:
  - Index finger up only        → Draw mode (draw with fingertip)
  - Index + Middle up           → Move mode (hover without drawing)
  - Fist (all fingers down)     → Erase last stroke
  - Show palm (all fingers up)  → Clear canvas
  - Pinch (thumb + index)       → Change color (cycles through palette)

Press 'q' to quit, 's' to save canvas.
"""

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_project_python_env():
    root = PROJECT_ROOT.resolve()

    env_roots = [
        root / "temp_env312",
        root / "venv312",
    ]

    python_candidates = []
    for env_root in env_roots:
        for python_name in ("python.exe", "python"):
            candidate = env_root / "Scripts" / python_name
            if candidate.exists():
                python_candidates.append(candidate)

    preferred_python = None
    for candidate in python_candidates:
        if "temp_env312" in str(candidate):
            preferred_python = candidate
            break
    if preferred_python is None and python_candidates:
        preferred_python = python_candidates[0]

    if __name__ == "__main__" and os.environ.get("_WHITEBOARD_REEXEC") != "1" and preferred_python is not None:
        current_exe = Path(sys.executable).resolve()
        if current_exe != preferred_python.resolve():
            env = os.environ.copy()
            env["_WHITEBOARD_REEXEC"] = "1"
            result = subprocess.run([str(preferred_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
            raise SystemExit(result.returncode)

    def _site_packages_for(env_root):
        for site_dir in (env_root / "Lib" / "site-packages", env_root / "lib" / "site-packages"):
            if site_dir.exists():
                return site_dir
        return None

    current_env_root = None
    current_exe = Path(sys.executable).resolve()
    for env_root in env_roots:
        if current_exe.is_relative_to(env_root.resolve()):
            current_env_root = env_root
            break

    chosen_site_packages = None
    if current_env_root is not None:
        chosen_site_packages = _site_packages_for(current_env_root)
    elif preferred_python is not None:
        chosen_env_root = preferred_python.parent.parent
        chosen_site_packages = _site_packages_for(chosen_env_root)

    if chosen_site_packages is not None and str(chosen_site_packages) not in sys.path:
        sys.path.insert(0, str(chosen_site_packages))


_ensure_project_python_env()

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from mediapipe import solutions as mp_solutions
    from mediapipe.solutions import drawing_utils as mp_drawing_utils
except ImportError:
    from mediapipe.python import solutions as mp_solutions
    from mediapipe.python.solutions import drawing_utils as mp_drawing_utils

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = [
    (0, 0, 255),      # Red
    (0, 165, 255),    # Orange
    (0, 255, 255),    # Yellow
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (255, 0, 255),    # Magenta
    (255, 255, 255),  # White
]
COLOR_NAMES = ["Red", "Orange", "Yellow", "Green", "Blue", "Magenta", "White"]

# ── Config ───────────────────────────────────────────────────────────────────
BRUSH_THICKNESS   = 6
ERASER_THICKNESS  = 40
SMOOTHING_WINDOW  = 5
GESTURE_COOLDOWN  = 0.8
WINDOW_NAME       = "Virtual Whiteboard"


# ── High-Quality Font Cache ──────────────────────────────────────────────────
class FontManager:
    """Loads system TrueType fonts with fallback for high-DPI text rendering."""
    _font_cache = {}

    @classmethod
    def get_font(cls, size: int, bold: bool = False):
        key = (size, bold)
        if key in cls._font_cache:
            return cls._font_cache[key]

        font_candidates = []
        if os.name == "nt":
            font_dir = Path("C:/Windows/Fonts")
            if bold:
                font_candidates.extend([font_dir / "segoeuib.ttf", font_dir / "arialbd.ttf"])
            font_candidates.extend([font_dir / "segoeui.ttf", font_dir / "arial.ttf", font_dir / "calibri.ttf"])
        elif sys.platform == "darwin":
            font_candidates.extend([
                Path("/System/Library/Fonts/SFNS.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
            ])
        else:
            font_candidates.extend([
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            ])

        font = None
        for font_path in font_candidates:
            if font_path.exists():
                try:
                    font = ImageFont.truetype(str(font_path), size=size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        cls._font_cache[key] = font
        return font


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.75, tracking_conf=0.75):
        self.mp_hands = mp_solutions.hands
        self.mp_drawing = mp_drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        return results

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(50, 50, 50), thickness=1),
                )

    def get_landmarks(self, results, frame_shape):
        if not results.multi_hand_landmarks:
            return None
        h, w = frame_shape[:2]
        lm = results.multi_hand_landmarks[0].landmark
        return [(int(p.x * w), int(p.y * h)) for p in lm]

    def fingers_up(self, landmarks):
        if landmarks is None:
            return [False] * 5

        tips  = [4, 8, 12, 16, 20]
        bases = [2, 6, 10, 14, 18]
        up    = []

        up.append(landmarks[tips[0]][0] < landmarks[bases[0]][0])
        for i in range(1, 5):
            up.append(landmarks[tips[i]][1] < landmarks[bases[i]][1])

        return up


class VirtualWhiteboard:
    def __init__(self, cam_index=0):
        self.cap, frame = self._open_capture(cam_index)
        self.h, self.w = frame.shape[:2]
        self.canvas    = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        self.tracker   = HandTracker()
        self.strokes   = []
        self.cur_stroke = []

        self.color_idx = 4
        self.smoothing_q = deque(maxlen=SMOOTHING_WINDOW)
        self.prev_point = None
        self.drawing = False
        self.last_gesture_time = 0

        self.gesture_msg = ""
        self.gesture_msg_time = 0

    def _open_capture(self, cam_index):
        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        candidates = [cam_index] if cam_index is not None else []
        candidates.extend([0, 1, 2, 3])

        seen = set()
        for index in candidates:
            if index in seen:
                continue
            seen.add(index)
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, frame
            cap.release()

        raise RuntimeError("Cannot open webcam. Verify device connection and permissions.")

    def _smooth_point(self, pt):
        self.smoothing_q.append(pt)
        xs = [p[0] for p in self.smoothing_q]
        ys = [p[1] for p in self.smoothing_q]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _gesture_cooldown_ok(self):
        return time.time() - self.last_gesture_time > GESTURE_COOLDOWN

    def _set_gesture(self, msg):
        self.last_gesture_time = time.time()
        self.gesture_msg = msg
        self.gesture_msg_time = time.time()

    def _commit_stroke(self):
        if self.cur_stroke:
            self.strokes.append(np.array(self.cur_stroke, dtype=object))
        self.cur_stroke = []

    def _undo_last_stroke(self):
        if self.strokes:
            self.strokes.pop()
            self._redraw_canvas()
            self._set_gesture("Undo last stroke")

    def _redraw_canvas(self):
        self.canvas[:] = 0
        for stroke in self.strokes:
            for i in range(1, len(stroke)):
                pt1, pt2, col, thick = stroke[i - 1]
                cv2.line(self.canvas, pt1, pt2, col, thick, lineType=cv2.LINE_AA)

    def _clear_canvas(self):
        self.canvas[:] = 0
        self.strokes.clear()
        self.cur_stroke = []
        self._set_gesture("Canvas cleared")

    def _detect_gesture(self, fingers, landmarks):
        thumb, index, middle, ring, pinky = fingers

        if all(fingers) and self._gesture_cooldown_ok():
            self._clear_canvas()
            return "clear"

        if not any(fingers) and self._gesture_cooldown_ok():
            self._undo_last_stroke()
            return "undo"

        if thumb and index and not middle and not ring and not pinky:
            if self._gesture_cooldown_ok():
                self.color_idx = (self.color_idx + 1) % len(COLORS)
                self._set_gesture(f"Color: {COLOR_NAMES[self.color_idx]}")
            return "color"

        if index and middle and ring and not pinky:
            return "eraser"

        if index and middle and not ring and not pinky:
            return "move"

        if index and not middle and not ring and not pinky:
            return "draw"

        return "idle"

    def _draw_on_canvas(self, point, mode):
        pt = self._smooth_point(point)

        if mode in {"draw", "eraser"}:
            if self.prev_point and self.drawing:
                col = (0, 0, 0) if mode == "eraser" else COLORS[self.color_idx]
                thick = ERASER_THICKNESS if mode == "eraser" else BRUSH_THICKNESS
                cv2.line(self.canvas, self.prev_point, pt, col, thick, lineType=cv2.LINE_AA)
                self.cur_stroke.append((self.prev_point, pt, col, thick))
            self.drawing = True
            self.prev_point = pt

        elif mode in ("move", "idle", "color", "clear", "undo"):
            if self.drawing:
                self._commit_stroke()
            self.drawing = False
            self.prev_point = None

    def _compose_frame(self, frame, canvas):
        return cv2.add(frame, canvas)

    def _draw_ui(self, frame, mode, fps, landmarks):
        fh, fw = frame.shape[:2]
        base_w = 1440.0
        s = max(0.70, min(1.60, fw / base_w))

        def S(v):
            return max(1, int(round(v * s)))

        BG       = (14, 16, 23)
        PANEL    = (23, 26, 35)
        PANEL2   = (31, 35, 47)
        BORDER   = (62, 68, 84)
        WHITE    = (245, 247, 252)
        MUTED    = (156, 163, 180)
        SUBTLE   = (110, 117, 135)
        ACCENT   = (218, 130, 255)
        GREEN    = (95, 220, 160)
        RED      = (110, 116, 238)
        YELLOW   = (95, 201, 244)
        BLUE     = (215, 150, 255)

        text_draw_queue = []

        def queue_text(text, pos, size_pt, bgr_color, bold=False):
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            text_draw_queue.append((text, pos, size_pt, rgb_color, bold))

        def rr(img, a, b, color, radius=12, thickness=-1):
            x1, y1 = a
            x2, y2 = b
            radius = max(1, min(radius, (x2-x1)//2, (y2-y1)//2))
            if thickness == -1:
                cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
                cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
                for px, py in (
                    (x1+radius, y1+radius), (x2-radius, y1+radius),
                    (x1+radius, y2-radius), (x2-radius, y2-radius)
                ):
                    cv2.circle(img, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        def panel(img, a, b, color=PANEL, radius=14, alpha=0.92):
            layer = img.copy()
            rr(layer, a, b, color, radius)
            cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)
            rr(img, a, b, BORDER, radius, max(1, S(1)))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), BG, -1)
        cv2.addWeighted(overlay, 0.045, frame, 0.955, 0, frame)

        # Header
        m = S(16)
        header_y2 = S(78)
        panel(frame, (m, m), (fw-m, header_y2), PANEL, S(16), 0.94)

        rr(frame, (S(29), S(29)), (S(65), S(65)), ACCENT, S(10))
        cv2.circle(frame, (S(47), S(47)), S(8), BG, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (S(47), S(47)), S(3), ACCENT, -1, lineType=cv2.LINE_AA)

        queue_text("AI Whiteboard", (S(80), S(26)), S(18), WHITE, bold=True)
        queue_text("Gesture-powered workspace", (S(81), S(50)), S(11), MUTED)

        # Status
        status_x = fw - S(185)
        cv2.circle(frame, (status_x, S(43)), S(5), GREEN, -1, lineType=cv2.LINE_AA)
        queue_text("CAMERA ON", (status_x + S(13), S(35)), S(11), WHITE, bold=True)
        queue_text(f"{max(0, fps):.0f} FPS", (fw - S(70), S(35)), S(11), MUTED)

        # Sidebar
        top = S(94)
        bottom = fh - S(18)
        sidebar_w = S(208)
        gap = S(18)

        sx1, sx2 = m, m + sidebar_w
        cx1, cx2 = sx2 + gap, fw - m

        panel(frame, (sx1, top), (sx2, bottom), PANEL, S(16), 0.94)
        queue_text("TOOLS", (sx1 + S(20), top + S(16)), S(11), SUBTLE, bold=True)

        tools = [
            ("Brush",   "draw",   GREEN),
            ("Eraser",  "eraser", RED),
            ("Move",    "move",   YELLOW),
            ("Undo",    "undo",   BLUE),
            ("Clear",   "clear",  (136, 140, 238)),
            ("Palette", "color",  ACCENT),
        ]

        tool_top = top + S(43)
        tool_h = S(43)
        tool_gap = S(7)

        for i, (name, tool_mode, color) in enumerate(tools):
            yy = tool_top + i * (tool_h + tool_gap)
            active = mode == tool_mode

            if active:
                rr(frame, (sx1+S(10), yy), (sx2-S(10), yy+tool_h), PANEL2, S(11))
                cv2.rectangle(frame, (sx1+S(10), yy+S(8)), (sx1+S(13), yy+tool_h-S(8)), color, -1)

            ix, iy = sx1 + S(34), yy + tool_h // 2

            if tool_mode == "draw":
                cv2.line(frame, (ix-S(7), iy+S(6)), (ix+S(7), iy-S(6)), color, S(2), cv2.LINE_AA)
            elif tool_mode == "eraser":
                cv2.rectangle(frame, (ix-S(8), iy-S(5)), (ix+S(8), iy+S(5)), color, S(2))
            elif tool_mode == "move":
                cv2.circle(frame, (ix, iy), S(7), color, S(2), cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), S(2), color, -1, cv2.LINE_AA)
            elif tool_mode == "undo":
                cv2.ellipse(frame, (ix, iy), (S(8), S(7)), 35, 35, 310, color, S(2), cv2.LINE_AA)
            elif tool_mode == "clear":
                cv2.rectangle(frame, (ix-S(7), iy-S(7)), (ix+S(7), iy+S(7)), color, S(2))
            else:
                cv2.circle(frame, (ix, iy), S(7), color, S(2), cv2.LINE_AA)

            queue_text(name, (sx1+S(57), yy+S(12)), S(12), WHITE if active else MUTED, bold=active)

        # Color Card
        card_h = S(72)
        card_y1 = bottom - card_h - S(12)
        rr(frame, (sx1+S(10), card_y1), (sx2-S(10), bottom-S(12)), PANEL2, S(12))
        queue_text("CURRENT COLOR", (sx1+S(22), card_y1+S(10)), S(9), SUBTLE, bold=True)
        cv2.circle(frame, (sx1+S(31), card_y1+S(47)), S(9), COLORS[self.color_idx], -1, cv2.LINE_AA)
        cv2.circle(frame, (sx1+S(31), card_y1+S(47)), S(11), BORDER, S(1), cv2.LINE_AA)
        queue_text(COLOR_NAMES[self.color_idx], (sx1+S(51), card_y1+S(38)), S(12), WHITE, bold=True)

        # Camera Stage Canvas
        panel(frame, (cx1, top), (cx2, bottom), (20, 23, 31), S(16), 0.35)
        cv2.rectangle(frame, (cx1+S(10), top+S(10)), (cx2-S(10), bottom-S(10)), (150, 157, 174), max(1, S(1)), cv2.LINE_AA)

        rr(frame, (cx1+S(16), top+S(15)), (cx1+S(172), top+S(51)), (20, 23, 31), S(10))
        queue_text("LIVE CANVAS", (cx1+S(28), top+S(18)), S(11), WHITE, bold=True)
        queue_text("Draw over your camera", (cx1+S(29), top+S(34)), S(9), MUTED)

        modes = {
            "draw":   ("DRAWING", GREEN),
            "eraser": ("ERASING", RED),
            "move":   ("MOVING", YELLOW),
            "color":  ("COLOR", ACCENT),
            "clear":  ("CLEARED", (136, 140, 238)),
            "undo":   ("UNDO", BLUE),
            "idle":   ("READY", (175, 181, 194)),
        }
        mode_name, mode_color = modes.get(mode, ("READY", MUTED))

        pill_w = S(112)
        pill_x = cx2 - pill_w - S(16)
        pill_y = top + S(16)
        rr(frame, (pill_x, pill_y), (pill_x+pill_w, pill_y+S(34)), (22, 25, 34), S(11))
        cv2.circle(frame, (pill_x+S(15), pill_y+S(17)), S(4), mode_color, -1, cv2.LINE_AA)
        queue_text(mode_name, (pill_x+S(27), pill_y+S(10)), S(10), WHITE, bold=True)

        # Floating Bottom Bar
        bar_h = S(58)
        bar_y = bottom - bar_h - S(14)
        bar_x1 = cx1 + S(14)
        bar_x2 = cx2 - S(14)

        rr(frame, (bar_x1, bar_y), (bar_x2, bar_y+bar_h), (19, 22, 30), S(13))
        rr(frame, (bar_x1, bar_y), (bar_x2, bar_y+bar_h), BORDER, S(13), max(1, S(1)))

        queue_text("COLOR", (bar_x1+S(16), bar_y+S(21)), S(10), SUBTLE, bold=True)

        swatch_start = bar_x1 + S(72)
        for i, col in enumerate(COLORS):
            xx = swatch_start + i*S(31)
            yy = bar_y + bar_h//2
            cv2.circle(frame, (xx, yy), S(9), col, -1, cv2.LINE_AA)
            if i == self.color_idx:
                cv2.circle(frame, (xx, yy), S(14), ACCENT, max(1, S(2)), cv2.LINE_AA)

        queue_text("Q", (bar_x2-S(145), bar_y+S(21)), S(10), WHITE, bold=True)
        queue_text("Quit", (bar_x2-S(128), bar_y+S(21)), S(10), MUTED)
        queue_text("S", (bar_x2-S(73), bar_y+S(21)), S(10), WHITE, bold=True)
        queue_text("Save", (bar_x2-S(57), bar_y+S(21)), S(10), MUTED)

        # Gesture Toast
        if self.gesture_msg and time.time() - self.gesture_msg_time < 1.7:
            msg = self.gesture_msg
            tw = S(min(360, max(200, 40 + len(msg) * 8)))
            toast_x = cx1 + S(20)
            toast_y = top + S(64)
            rr(frame, (toast_x, toast_y), (toast_x+tw, toast_y+S(38)), (25, 28, 38), S(12))
            rr(frame, (toast_x, toast_y), (toast_x+tw, toast_y+S(38)), BORDER, S(12), max(1, S(1)))
            cv2.circle(frame, (toast_x+S(15), toast_y+S(19)), S(4), ACCENT, -1, cv2.LINE_AA)
            queue_text(msg, (toast_x+S(27), toast_y+S(11)), S(11), WHITE)

        # Cursor
        if landmarks:
            tip = landmarks[8]
            if mode == "draw":
                cv2.circle(frame, tip, S(12), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tip, S(8), COLORS[self.color_idx], -1, cv2.LINE_AA)
                cv2.circle(frame, tip, S(12), ACCENT, max(1, S(1)), cv2.LINE_AA)
            elif mode == "eraser":
                cv2.circle(frame, tip, S(22), (238, 241, 246), -1, cv2.LINE_AA)
                cv2.circle(frame, tip, S(27), RED, max(1, S(2)), cv2.LINE_AA)
            else:
                cv2.circle(frame, tip, S(7), COLORS[self.color_idx], -1, cv2.LINE_AA)
                cv2.circle(frame, tip, S(11), (255, 255, 255), max(1, S(2)), cv2.LINE_AA)

        # Render all queued high-res TrueType text via Pillow
        if text_draw_queue:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            draw = ImageDraw.Draw(pil_img)
            for text, (tx, ty), pt_size, rgb_color, bold in text_draw_queue:
                font = FontManager.get_font(pt_size, bold=bold)
                draw.text((tx, ty), text, font=font, fill=rgb_color)
            frame[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def run(self):
        print("Virtual Whiteboard running. Press 'q' to quit, 's' to save.")
        prev_time = time.time()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        initial_w = min(max(self.w, 1280), 1600)
        initial_h = int(initial_w * self.h / self.w)
        if initial_h > 950:
            initial_h = 950
            initial_w = int(initial_h * self.w / self.h)
        cv2.resizeWindow(WINDOW_NAME, initial_w, initial_h)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                results = self.tracker.process(frame)
                landmarks = self.tracker.get_landmarks(results, frame.shape)
                fingers = self.tracker.fingers_up(landmarks)
                mode = self._detect_gesture(fingers, landmarks)

                if landmarks:
                    fingertip = landmarks[8]
                    self._draw_on_canvas(fingertip, mode)
                else:
                    if self.drawing:
                        self._commit_stroke()
                    self.drawing = False
                    self.prev_point = None

                self.tracker.draw_landmarks(frame, results)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                self._draw_ui(frame, mode, fps, landmarks)

                combined = self._compose_frame(frame, self.canvas)
                cv2.imshow(WINDOW_NAME, combined)       

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    fname = f"whiteboard_{int(time.time())}.png"
                    save_path = Path(fname)
                    save_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(save_path), self.canvas)
                    self._set_gesture(f"Saved {save_path.name}")
                    print(f"Canvas saved -> {save_path}")

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Virtual Whiteboard")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera index to use (default: 0)")
    args = parser.parse_args()

    try:
        board = VirtualWhiteboard(cam_index=args.cam_index)
        board.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except RuntimeError as exc:
        print(f"Unable to start Virtual Whiteboard: {exc}")
        sys.exit(1)