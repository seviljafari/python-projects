import os
import sys
import time
import argparse
import urllib.request
from datetime import datetime


def fatal(title, en, fa=""):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    text = en + (("\n\n" + fa) if fa else "")
    print("\n[Air Canvas] ERROR: " + title + "\n" + text + "\n", file=sys.stderr)
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, "Air Canvas - " + title, 0x10)
        except Exception:
            pass
    sys.exit(1)


INSTALL_CMD = "pip install opencv-python numpy mediapipe==0.10.14"

try:
    import numpy as np
except Exception as e:
    fatal("NumPy is not installed",
          "NumPy could not be imported (%s).\nInstall the requirements with:\n  %s" % (e, INSTALL_CMD),
          "کتابخانه NumPy نصب نیست. دستور نصب:\n  " + INSTALL_CMD)

try:
    import cv2
except Exception as e:
    fatal("OpenCV is not installed",
          "OpenCV could not be imported (%s).\nInstall the requirements with:\n  %s" % (e, INSTALL_CMD),
          "کتابخانه OpenCV نصب نیست. دستور نصب:\n  " + INSTALL_CMD)


WIN = "Air Canvas"

PALETTE = [
    ("Purple", (247, 85, 168)),
    ("Yellow", (21, 204, 250)),
    ("Green",  (94, 197, 34)),
    ("Pink",   (153, 72, 236)),
]

IDLE, DRAW, ERASE, GRAB = 0, 1, 2, 3
MODE_LABEL = {IDLE: "READY", DRAW: "DRAWING", ERASE: "ERASER", GRAB: "GRAB / MOVE"}
MODE_COLOR = {IDLE: (170, 170, 170), DRAW: (120, 255, 160),
              ERASE: (120, 190, 255), GRAB: (90, 230, 255)}

CONFIRM = {IDLE: 2, DRAW: 4, ERASE: 3, GRAB: 3}
LOST_FRAMES = 5
UP_RATIO = 1.12
PINCH_ON, PINCH_OFF = 0.30, 0.55
COOLDOWN = 0.35
ERASER_RADIUS = 30
GRAB_RADIUS = 48
MIN_STEP = 1.5
POINT_SPACING = 3.0
JUMP_LIMIT_FRAC = 0.22

PANEL_X0, PANEL_X1 = 12, 104
BTN_CX = (PANEL_X0 + PANEL_X1) // 2
PANEL_TITLE_H = 36
BTN_CELL = 76
PANEL_H = PANEL_TITLE_H + BTN_CELL * len(PALETTE) + 10

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGER_JOINTS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1, shadow=True):
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def text_width(text, scale=0.6, thick=1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]


def rounded_rect(img, p1, p2, r, color, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        t = thickness
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, t, cv2.LINE_AA)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Stroke:
    __slots__ = ("pts", "color", "thick", "lo", "hi")

    def __init__(self, pts, color, thick):
        self.pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2).copy()
        self.color = tuple(int(c) for c in color)
        self.thick = int(thick)
        self.lo = self.pts.min(axis=0)
        self.hi = self.pts.max(axis=0)

    def translate(self, d):
        d = np.asarray(d, dtype=np.float32)
        self.pts += d
        self.lo = self.lo + d
        self.hi = self.hi + d

    def distance_to(self, p):
        d2 = ((self.pts - np.asarray(p, np.float32)) ** 2).sum(axis=1)
        return float(np.sqrt(d2.min()))


def draw_stroke(img, pts, color, thick):
    ip = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32).reshape(-1, 2)
    r = max(1, thick // 2)
    if len(ip) == 1:
        cv2.circle(img, (int(ip[0][0]), int(ip[0][1])), r, color, -1, cv2.LINE_AA)
        return
    cv2.polylines(img, [ip.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)
    cv2.circle(img, (int(ip[0][0]), int(ip[0][1])), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(ip[-1][0]), int(ip[-1][1])), r, color, -1, cv2.LINE_AA)


class Smoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.p = None

    def reset(self, p=None):
        self.p = None if p is None else np.asarray(p, dtype=np.float32).copy()

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float32)
        if self.p is None:
            self.p = p.copy()
        else:
            self.p = self.p + self.alpha * (p - self.p)
        return self.p.copy()


class AirCanvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.strokes = []
        self.active = None
        self.active_t0 = 0.0
        self.color_idx = 0
        self.thick = 8

        self.mode = IDLE
        self.raw = IDLE
        self.cand, self.cand_n = IDLE, 0
        self.pinching = False
        self.missing = 0
        self.block_until = 0.0

        self.smooth = Smoother(0.6)
        self.grab = None
        self.grab_last = None
        self.erase_last = None

        self.hand = None
        self.pointer = None

        self.dirty = True
        self.static = np.zeros((h, w, 3), np.uint8)
        self.toast_text, self.toast_until = "", 0.0
        self.mouse = (-1, -1)

        self.panel_y0 = max(10, (h - PANEL_H) // 2)
        self.panel_y1 = self.panel_y0 + PANEL_H
        mask = np.zeros((h, w), np.uint8)
        rounded_rect(mask, (PANEL_X0, self.panel_y0), (PANEL_X1, self.panel_y1), 20, 255, -1)
        self.panel_mask = (mask.astype(np.float32) / 255.0 * 0.80)[..., None]
        self.panel_rect = (PANEL_X0, self.panel_y0, PANEL_X1, self.panel_y1)

    @property
    def color(self):
        return PALETTE[self.color_idx][1]

    def show_toast(self, text, seconds=2.0):
        self.toast_text = text
        self.toast_until = time.time() + seconds

    def button_center(self, i):
        return BTN_CX, self.panel_y0 + PANEL_TITLE_H + BTN_CELL * i + BTN_CELL // 2

    def in_panel(self, p, margin=14):
        return (PANEL_X0 - margin <= p[0] <= PANEL_X1 + margin and
                self.panel_y0 - margin <= p[1] <= self.panel_y1 + margin)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(len(PALETTE)):
                cx, cy = self.button_center(i)
                if abs(x - cx) <= 38 and abs(y - cy) <= 36:
                    self.color_idx = i
                    self.show_toast("Color: " + PALETTE[i][0], 1.2)
                    return

    def clear(self):
        self.strokes = []
        self.active = None
        self.grab = None
        self.dirty = True
        self.show_toast("Canvas cleared")

    def render_clean(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        for s in self.strokes:
            draw_stroke(img, s.pts, s.color, s.thick)
        if self.active:
            draw_stroke(img, self.active, self.color, self.thick)
        return img

    def save(self, folder=None):
        folder = folder or os.path.dirname(os.path.abspath(__file__))
        name = datetime.now().strftime("air_canvas_%Y%m%d_%H%M%S.png")
        path = os.path.join(folder, name)
        ok, buf = cv2.imencode(".png", self.render_clean())
        if not ok:
            self.show_toast("Save failed!", 2.5)
            return None
        try:
            buf.tofile(path)
        except Exception:
            try:
                path = os.path.join(os.getcwd(), name)
                buf.tofile(path)
            except Exception as e:
                self.show_toast("Save failed: %s" % e, 3)
                return None
        self.show_toast("Saved: " + name, 3)
        print("[Air Canvas] saved ->", path)
        return path

    def finish_stroke(self, now):
        pts = self.active
        self.active = None
        if not pts:
            return
        if len(pts) < 2 and (now - self.active_t0) < 0.35:
            return
        self.strokes.append(Stroke(pts, self.color, self.thick))
        self.dirty = True

    def step_draw(self, p, now):
        p = np.asarray(p, np.float32)
        if not self.active:
            self.active = [p]
            self.active_t0 = now
            return
        last = np.asarray(self.active[-1], np.float32)
        d = dist(p, last)
        if d > JUMP_LIMIT_FRAC * self.w:
            self.finish_stroke(now)
            self.active = [p]
            self.active_t0 = now
            return
        if d < MIN_STEP:
            return
        n = int(d // POINT_SPACING) + 1
        for k in range(1, n + 1):
            self.active.append(last + (p - last) * (k / n))

    def erase_at(self, c):
        c = np.asarray(c, np.float32)
        r = ERASER_RADIUS
        new_list = []
        changed = False
        for s in self.strokes:
            if (c[0] < s.lo[0] - r or c[0] > s.hi[0] + r or
                    c[1] < s.lo[1] - r or c[1] > s.hi[1] + r):
                new_list.append(s)
                continue
            d2 = ((s.pts - c) ** 2).sum(axis=1)
            hit = d2 <= r * r
            if not hit.any():
                new_list.append(s)
                continue
            changed = True
            idx = np.flatnonzero(~hit)
            if len(idx) == 0:
                continue
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for run in runs:
                if len(run) >= 2:
                    new_list.append(Stroke(s.pts[run], s.color, s.thick))
        if changed:
            self.strokes = new_list
            self.dirty = True

    def step_erase(self, p):
        p = np.asarray(p, np.float32)
        if self.erase_last is None:
            self.erase_at(p)
        else:
            d = dist(p, self.erase_last)
            n = int(d // (ERASER_RADIUS * 0.5)) + 1
            for k in range(1, n + 1):
                self.erase_at(self.erase_last + (p - self.erase_last) * (k / n))
        self.erase_last = p

    def find_stroke(self, p):
        best, best_d = None, 1e9
        for s in self.strokes:
            if (p[0] < s.lo[0] - GRAB_RADIUS or p[0] > s.hi[0] + GRAB_RADIUS or
                    p[1] < s.lo[1] - GRAB_RADIUS or p[1] > s.hi[1] + GRAB_RADIUS):
                continue
            d = s.distance_to(p) - s.thick / 2.0
            if d < best_d:
                best, best_d = s, d
        return best if best_d <= GRAB_RADIUS else None

    def begin_grab(self, p):
        self.grab = self.find_stroke(p)
        self.grab_last = np.asarray(p, np.float32)
        if self.grab is not None:
            self.strokes.remove(self.grab)
            self.strokes.append(self.grab)
        self.dirty = True

    def step_grab(self, p):
        p = np.asarray(p, np.float32)
        if self.grab is not None and self.grab_last is not None:
            self.grab.translate(p - self.grab_last)
            m = 24
            s = self.grab
            sx = sy = 0.0
            if s.hi[0] < m:
                sx = m - s.hi[0]
            elif s.lo[0] > self.w - m:
                sx = (self.w - m) - s.lo[0]
            if s.hi[1] < m:
                sy = m - s.hi[1]
            elif s.lo[1] > self.h - m:
                sy = (self.h - m) - s.lo[1]
            if sx or sy:
                s.translate((sx, sy))
            self.dirty = True
        self.grab_last = p

    def release_grab(self):
        self.grab = None
        self.grab_last = None
        self.dirty = True

    def set_mode(self, new, target, now):
        old = self.mode
        if old == DRAW:
            self.finish_stroke(now)
        elif old == GRAB:
            self.release_grab()
            self.block_until = now + COOLDOWN
        elif old == ERASE:
            self.erase_last = None
            self.block_until = now + COOLDOWN
        self.mode = new
        self.smooth.reset(target)
        if new == GRAB and target is not None:
            self.begin_grab(target)

    def classify(self, pts):
        wrist = pts[0]
        palm = max(dist(pts[0], pts[9]), 1.0)
        ratio = dist(pts[4], pts[8]) / palm
        reach = dist(pts[8], wrist) / palm
        close = ratio < (PINCH_OFF if self.pinching else PINCH_ON)
        self.pinching = close and reach > 1.0

        up = {}
        for name, (tip, pip) in FINGER_JOINTS.items():
            up[name] = dist(pts[tip], wrist) > UP_RATIO * dist(pts[pip], wrist)

        targets = {
            DRAW: pts[8], ERASE: pts[20], IDLE: pts[8],
            GRAB: (pts[4] + pts[8]) / 2.0,
        }
        if self.pinching:
            raw = GRAB
        elif up["pinky"] and not up["index"] and not up["middle"]:
            raw = ERASE
        elif up["index"] and not up["middle"] and not up["ring"] and not up["pinky"]:
            raw = DRAW
        else:
            raw = IDLE
        return raw, targets

    def update(self, pts, now=None):
        now = time.time() if now is None else now
        self.hand = pts
        if pts is None:
            self.missing += 1
            self.pointer = None
            self.raw = IDLE
            if self.missing > LOST_FRAMES:
                if self.mode != IDLE:
                    self.set_mode(IDLE, None, now)
                self.cand, self.cand_n = IDLE, 0
                self.pinching = False
            return

        self.missing = 0
        raw, targets = self.classify(pts)
        if raw == DRAW and self.in_panel(pts[8]):
            raw = IDLE
        if raw == DRAW and now < self.block_until:
            raw = IDLE
        self.raw = raw

        if raw == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = raw, 1

        if self.mode == GRAB and raw != GRAB:
            self.set_mode(IDLE, None, now)
        if raw != self.mode and self.cand_n >= CONFIRM[raw]:
            self.set_mode(raw, targets[raw], now)

        if raw == self.mode:
            p = self.smooth(targets[raw])
            self.pointer = p
            if self.mode == DRAW:
                self.step_draw(p, now)
            elif self.mode == ERASE:
                self.step_erase(p)
            elif self.mode == GRAB:
                self.step_grab(p)
        else:
            self.smooth.reset()
            self.pointer = np.asarray(targets[raw], np.float32)

    def render_static(self):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        for s in self.strokes:
            if s is self.grab:
                glow = tuple(min(255, int(c * 0.5 + 130)) for c in s.color)
                draw_stroke(img, s.pts, glow, s.thick + 10)
            draw_stroke(img, s.pts, s.color, s.thick)
        self.static = img
        self.dirty = False

    def draw_palette(self, img):
        x0, y0, x1, y1 = self.panel_rect
        roi = img[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        m = self.panel_mask[y0:y1 + 1, x0:x1 + 1]
        bg = np.array((34, 30, 30), np.float32)
        img[y0:y1 + 1, x0:x1 + 1] = (roi * (1 - m) + bg * m).astype(np.uint8)
        rounded_rect(img, (x0, y0), (x1, y1), 20, (90, 85, 85), 1)
        t = "COLORS"
        put_text(img, t, ((x0 + x1) // 2 - text_width(t, 0.45) // 2, y0 + 24),
                 0.45, (200, 200, 200), 1)
        for i, (_, col) in enumerate(PALETTE):
            cx, cy = self.button_center(i)
            hover = abs(self.mouse[0] - cx) <= 38 and abs(self.mouse[1] - cy) <= 36
            sel = (i == self.color_idx)
            if sel:
                cv2.circle(img, (cx, cy), 36, tuple(int(c * 0.45) for c in col), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 34, (255, 255, 255), 3, cv2.LINE_AA)
            elif hover:
                cv2.circle(img, (cx, cy), 33, (150, 150, 150), 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 27, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 27, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
            if sel:
                cv2.circle(img, (cx - 8, cy - 9), 5, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_hand(self, img):
        pts = self.hand
        if pts is None:
            return
        ip = np.round(pts).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(ip[a]), tuple(ip[b]), (215, 215, 215), 2, cv2.LINE_AA)
        for i, p in enumerate(ip):
            tip = i in (4, 8, 12, 16, 20)
            cv2.circle(img, tuple(p), 6 if tip else 4, (255, 255, 255) if tip else (190, 190, 190),
                       -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 6 if tip else 4, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(ip[8]), 7, self.color, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(ip[8]), 7, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_pointer(self, img):
        if self.hand is None or self.pointer is None:
            return
        p = (int(self.pointer[0]), int(self.pointer[1]))
        if self.raw == DRAW:
            cv2.circle(img, p, self.thick // 2 + 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, p, max(2, self.thick // 2), self.color, -1, cv2.LINE_AA)
        elif self.raw == ERASE:
            ov = img.copy()
            cv2.circle(ov, p, ERASER_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
            cv2.circle(img, p, ERASER_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.raw == GRAB:
            a = tuple(np.round(self.hand[4]).astype(int))
            b = tuple(np.round(self.hand[8]).astype(int))
            col = (90, 230, 255)
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
            has = self.grab is not None if self.mode == GRAB else (self.find_stroke(self.pointer) is not None)
            ring = (120, 255, 140) if has else (110, 110, 255)
            cv2.circle(img, p, 14, ring, 3, cv2.LINE_AA)
            cv2.circle(img, p, GRAB_RADIUS, ring, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 12, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_hud(self, img, fps):
        label = MODE_LABEL[self.raw if self.hand is not None else IDLE]
        if self.hand is None:
            label, col = "SHOW YOUR HAND", (140, 140, 140)
        else:
            col = MODE_COLOR[self.raw]
        tw = text_width(label, 0.7, 2)
        cx = self.w // 2
        rounded_rect(img, (cx - tw // 2 - 22, 14), (cx + tw // 2 + 22, 52), 18, (30, 28, 28), -1)
        rounded_rect(img, (cx - tw // 2 - 22, 14), (cx + tw // 2 + 22, 52), 18, col, 1)
        put_text(img, label, (cx - tw // 2, 40), 0.7, col, 2, shadow=False)

        info = "FPS %d   Brush %d" % (fps, self.thick)
        put_text(img, info, (self.w - text_width(info, 0.5) - 14, 28), 0.5, (150, 150, 150))

        help_t = "Index: draw   Pinch: grab & move   Pinky: erase   |   C clear   S save   +/- size   Q/Esc quit"
        scale = 0.5 if self.w >= 1000 else 0.4
        put_text(img, help_t, (self.w // 2 - text_width(help_t, scale) // 2, self.h - 14),
                 scale, (135, 135, 135))

        if time.time() < self.toast_until:
            tw = text_width(self.toast_text, 0.65, 2)
            y = self.h - 60
            rounded_rect(img, (cx - tw // 2 - 18, y - 28), (cx + tw // 2 + 18, y + 12), 16, (40, 36, 36), -1)
            put_text(img, self.toast_text, (cx - tw // 2, y), 0.65, (255, 255, 255), 2, shadow=False)

    def compose(self, fps=0):
        if self.dirty:
            self.render_static()
        img = self.static.copy()
        if self.active:
            draw_stroke(img, self.active, self.color, self.thick)
        self.draw_palette(img)
        self.draw_hand(img)
        self.draw_pointer(img)
        self.draw_hud(img, fps)
        return img


class HandTracker:
    def __init__(self):
        try:
            import mediapipe as mp
        except Exception as e:
            fatal("MediaPipe is not installed",
                  "MediaPipe could not be imported (%s).\nInstall the requirements with:\n  %s"
                  % (e, INSTALL_CMD),
                  "کتابخانه MediaPipe نصب نیست. دستور نصب:\n  " + INSTALL_CMD)
        self.mp = mp
        self.legacy = None
        self.landmarker = None
        self.t0 = time.time()
        self.last_ts = -1

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.legacy = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.6)
        else:
            self._init_tasks()

    def _init_tasks(self):
        mp = self.mp
        try:
            from mediapipe.tasks.python import vision, BaseOptions
        except Exception as e:
            fatal("Unsupported MediaPipe version",
                  "This MediaPipe build has neither the classic nor the Tasks hand API (%s).\n"
                  "Install a compatible version:\n  %s" % (e, INSTALL_CMD),
                  "نسخه‌ی MediaPipe سازگار نیست. دستور نصب:\n  " + INSTALL_CMD)
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        if not os.path.isfile(model):
            print("[Air Canvas] downloading hand model (one time, ~8 MB) ...")
            try:
                urllib.request.urlretrieve(MODEL_URL, model)
            except Exception as e:
                try:
                    if os.path.isfile(model):
                        os.remove(model)
                except Exception:
                    pass
                fatal("Could not download the hand model",
                      "Your MediaPipe version needs the file 'hand_landmarker.task' but the download "
                      "failed (%s).\nEither connect to the internet and run again, or install the "
                      "version that needs no download:\n  %s\n(or download it manually from\n  %s\n"
                      "and put it next to air_canvas.py)" % (e, INSTALL_CMD, MODEL_URL),
                      "دانلود مدل تشخیص دست ناموفق بود. اینترنت را وصل کنید یا دستور زیر را اجرا کنید:\n  "
                      + INSTALL_CMD)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def process(self, bgr, w, h):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = None
        if self.legacy is not None:
            rgb.flags.writeable = False
            res = self.legacy.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
        else:
            ts = int((time.time() - self.t0) * 1000)
            if ts <= self.last_ts:
                ts = self.last_ts + 1
            self.last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.landmarker.detect_for_video(img, ts)
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
        if lm is None:
            return None
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self):
        try:
            if self.legacy is not None:
                self.legacy.close()
            if self.landmarker is not None:
                self.landmarker.close()
        except Exception:
            pass


def try_open(index, width, height):
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
            time.sleep(0.05)
        cap.release()
    return None, None


def open_camera(index, width, height):
    cap, frame = try_open(index, width, height)
    if cap is not None:
        return cap, frame
    for alt in range(0, 4):
        if alt == index:
            continue
        cap, frame = try_open(alt, width, height)
        if cap is not None:
            print("[Air Canvas] camera %d not available, using camera %d" % (index, alt))
            return cap, frame
    fatal("Camera could not be opened",
          "No working webcam was found.\n"
          "- Make sure a webcam is connected and not used by another program (Zoom, Teams, browser ...).\n"
          "- Windows: Settings > Privacy & security > Camera > allow desktop apps to access the camera.\n"
          "- Try another camera index:  python air_canvas.py --camera 1",
          "دوربین باز نشد. مطمئن شوید وب‌کم وصل است، برنامه‌ی دیگری از آن استفاده نمی‌کند "
          "و در تنظیمات ویندوز (Privacy > Camera) دسترسی برنامه‌ها به دوربین فعال است. "
          "می‌توانید دوربین دیگری را امتحان کنید:  python air_canvas.py --camera 1")


def main():
    ap = argparse.ArgumentParser(description="Air Canvas - draw in the air with your hand")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280, help="requested capture width")
    ap.add_argument("--height", type=int, default=720, help="requested capture height")
    args = ap.parse_args()

    cap, first = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker()

    h, w = first.shape[:2]
    app = AirCanvas(w, h)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, app.on_mouse)
    print("[Air Canvas] running at %dx%d  -  press Q or Esc to quit" % (w, h))

    fps, t_prev, bad_reads = 0.0, time.time(), 0
    frame = first
    try:
        while True:
            if frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    bad_reads += 1
                    frame = None
                    if bad_reads > 60:
                        fatal("Camera stopped working",
                              "The camera stopped delivering frames. Reconnect it and start again.",
                              "دوربین دیگر تصویر نمی‌دهد. آن را دوباره وصل کنید و برنامه را اجرا کنید.")
                    time.sleep(0.02)
                    continue
            bad_reads = 0
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            frame = cv2.flip(frame, 1)

            pts = tracker.process(frame, w, h)
            now = time.time()
            app.update(pts, now)

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            cv2.imshow(WIN, app.compose(int(round(fps))))
            frame = None

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("c"), ord("C")):
                app.clear()
            elif key in (ord("s"), ord("S")):
                app.save()
            elif key in (ord("+"), ord("=")):
                app.thick = min(40, app.thick + 2)
            elif key in (ord("-"), ord("_")):
                app.thick = max(2, app.thick - 2)

            try:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()