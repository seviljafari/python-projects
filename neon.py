import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

SPAN_MIN = 0.10
SPAN_MAX = 0.85
ACTIVE_LIMIT = 0.95

vol_percent = 50
last_vol = -1

YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
SKELETON_DIM = (40, 60, 70)
WHITE = (240, 250, 250)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def map_span(s):
    return clamp((s - SPAN_MIN) / (SPAN_MAX - SPAN_MIN) * 100.0, 0, 100)

def lerp_color(c1, c2, t):
    t = clamp(t, 0.0, 1.0)
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))

def gradient_color_4(t):
    t = clamp(t, 0.0, 1.0)
    if t < 0.33:
        return lerp_color(YELLOW, ORANGE, t / 0.33)
    elif t < 0.66:
        return lerp_color(ORANGE, GREEN, (t - 0.33) / 0.33)
    else:
        return lerp_color(GREEN, RED, (t - 0.66) / 0.34)

def neon_line(img, p1, p2, color, thickness, glow=True):
    if glow:
        glow_layer = np.zeros_like(img)
        cv2.line(glow_layer, p1, p2, color, thickness + 8, cv2.LINE_AA)
        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), 7)
        img[:] = cv2.addWeighted(img, 1.0, glow_layer, 0.6, 0)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def neon_circle(img, center, radius, color, thickness):
    glow_layer = np.zeros_like(img)
    cv2.circle(glow_layer, center, radius + 6, color, thickness + 4, cv2.LINE_AA)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), 6)
    img[:] = cv2.addWeighted(img, 1.0, glow_layer, 0.6, 0)
    cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    active = None
    vol_val = vol_percent

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]

        for a, b in mp_hands.HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], SKELETON_DIM, 2, cv2.LINE_AA)

        for i, (x, y) in enumerate(pts):
            if i in (4, 8):
                cv2.circle(canvas, (x, y), 6, WHITE, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 9, (80, 90, 95), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), 3, (180, 200, 200), -1, cv2.LINE_AA)

        thumb, idx = pts[4], pts[8]
        palm = math.dist(pts[5], pts[17])
        if palm > 1:
            di = math.dist(thumb, idx) / palm
            if di < ACTIVE_LIMIT:
                active = "volume"
                vol_val = map_span(di)

        neon_line(canvas, thumb, idx, YELLOW,
                  5 if active == "volume" else 2)

        if active == "volume":
            neon_circle(canvas, idx, 10, YELLOW, 2)

    if active == "volume":
        t = int(round(vol_val))
        if abs(t - last_vol) >= 2:
            try:
                vol_target = min_vol + (max_vol - min_vol) * (t / 100.0)
                volume.SetMasterVolumeLevel(vol_target, None)
                last_vol = t
            except Exception as e:
                print(f"Volume error: {e}")

    vol_percent = vol_val

    bar_x1, bar_x2 = 25, 55
    bar_y1, bar_y2 = 90, h - 90
    bar_w = bar_x2 - bar_x1
    bar_h = bar_y2 - bar_y1

    cv2.rectangle(canvas, (bar_x1 - 5, bar_y1 - 5),
                  (bar_x2 + 5, bar_y2 + 5), (18, 18, 18), -1)

    glow_border = np.zeros_like(canvas)
    cv2.rectangle(glow_border, (bar_x1 - 5, bar_y1 - 5),
                  (bar_x2 + 5, bar_y2 + 5), RED, 3, cv2.LINE_AA)
    glow_border = cv2.GaussianBlur(glow_border, (0, 0), 12)
    canvas = cv2.addWeighted(canvas, 1.0, glow_border, 1.0, 0)
    cv2.rectangle(canvas, (bar_x1 - 5, bar_y1 - 5),
                  (bar_x2 + 5, bar_y2 + 5), (255, 180, 180), 2, cv2.LINE_AA)

    fill_h = int(bar_h * vol_percent / 100.0)
    if fill_h > 0:
        for i in range(fill_h):
            t = i / bar_h
            c = gradient_color_4(t)
            y = bar_y2 - i
            cv2.line(canvas, (bar_x1, y), (bar_x2, y), c, 1)

    for frac in [0.25, 0.5, 0.75]:
        y_guide = int(bar_y2 - bar_h * frac)
        cv2.line(canvas, (bar_x1, y_guide), (bar_x2, y_guide),
                 (45, 45, 45), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{int(vol_percent)}%", (bar_x1 - 8, bar_y1 - 14),
                cv2.FONT_HERSHEY_PLAIN, 1.8, RED, 2)

    cv2.putText(canvas, "VOL", (bar_x1 + 2, bar_y2 + 28),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (210, 220, 220), 2)

    marker_y = int(bar_y2 - bar_h * vol_percent / 100.0)
    triangle = np.array([
        [bar_x2 + 12, marker_y - 7],
        [bar_x2 + 12, marker_y + 7],
        [bar_x2 + 24, marker_y]
    ], np.int32)
    cv2.fillPoly(canvas, [triangle], RED, cv2.LINE_AA)

    cv2.imshow("Hand Volume Control", canvas)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()