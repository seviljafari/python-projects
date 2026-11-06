import math
import cv2
import mediapipe as mp
import numpy as np

# سطوح مختلف تقارن (چند بار دور مرکز تکرار شود)
SYMMETRY_LEVELS = [1, 4, 6, 8, 12]

# Keys:
#   m -> change symmetry  (1 / 4 / 6 / 8 / 12 segments)
#   n -> neon glow ON/OFF
#   r -> rainbow color ON/OFF
#   c -> clear canvas
#   q -> quit

def sym_points(x, y, cx, cy, n, mirror):
    # محاسبه نقاط متقارن حول مرکز (چرخش و ایجاد آینه)
    dx, dy = x - cx, y - cy
    points = []
    for k in range(n):
        a = 2 * math.pi * k / n
        cos_a, sin_a = math.cos(a), math.sin(a)
        points.append((int(cx + dx * cos_a - dy * sin_a),
                       int(cy + dx * sin_a + dy * cos_a)))
        if mirror:
            mx = -dx
            points.append((int(cx + mx * cos_a - dy * sin_a),
                           int(cy + mx * sin_a + dy * cos_a)))
    return points


def rainbow_color(t):
    # تولید رنگ رنگین‌کمانی بر اساس زمان
    hsv = np.uint8([[[(t * 2) % 180, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def render(frame, canvas, neon):
    # ترکیب تصویر وبکم با بوم نقاشی
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255

    if not neon:
        inv = cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_or(cv2.bitwise_and(frame, inv), canvas)

    # اعمال افکت نئون و درخشش در صورت فعال بودن
    h, w = canvas.shape[:2]
    out = cv2.convertScaleAbs(frame, alpha=0.4)
    small = cv2.resize(canvas, (w // 4, h // 4))
    glow = cv2.GaussianBlur(small, (0, 0), 7)
    glow = cv2.resize(glow, (w, h))
    out = cv2.add(out, cv2.convertScaleAbs(glow, alpha=1.8))
    inv = cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR)
    out = cv2.bitwise_or(cv2.bitwise_and(out, inv), canvas)
    core = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    out[core] = np.clip(out[core] * 0.35 + 255 * 0.65, 0, 255).astype(np.uint8)
    return out


def main():
    # راه‌اندازی MediaPipe برای تشخیص دست
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # راه‌اندازی وبکم
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    canvas = None
    prev = None

    # تنظیمات اولیه قلم و پاک‌کن
    draw_color = (255, 255, 0)
    brush_thickness = 10
    eraser_thickness = 50

    # متغیرهای وضعیت برنامه (تقارن، نئون، رنگین‌کمان)
    sym_index = 0
    neon = False
    rainbow = False
    tick = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        tick += 1

        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        n = SYMMETRY_LEVELS[sym_index]
        mirror = n > 1
        color = rainbow_color(tick) if rainbow else draw_color

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = [[int(p.x * w), int(p.y * h)] for p in hand_landmarks.landmark]

                x_i, y_i = lm[8]
                x_p, y_p = lm[20]
                index_up = lm[8][1] < lm[6][1]
                pinky_up = lm[20][1] < lm[18][1]

                # حالت پاک‌کن (بلند کردن انگشت کوچک)
                if pinky_up:
                    prev = None
                    for px, py in sym_points(x_p, y_p, cx, cy, n, mirror):
                        cv2.circle(frame, (px, py), 25, (0, 0, 255), 2)
                        cv2.circle(canvas, (px, py), eraser_thickness, (0, 0, 0), -1)

                # حالت نقاشی (بلند کردن انگشت اشاره)
                elif index_up:
                    pts = sym_points(x_i, y_i, cx, cy, n, mirror)
                    for p in pts:
                        cv2.circle(frame, p, 8, color, cv2.FILLED)

                    if prev is not None:
                        prev_pts = sym_points(prev[0], prev[1], cx, cy, n, mirror)
                        for a, b in zip(prev_pts, pts):
                            cv2.line(canvas, a, b, color, brush_thickness)
                    prev = (x_i, y_i)

                else:
                    prev = None
        else:
            prev = None

        output = render(frame, canvas, neon)

        # نمایش نقطه مرکزی تقارن
        if n > 1:
            cv2.circle(output, (cx, cy), 4, (255, 255, 255), -1)

        # نمایش وضعیت فعلی روی صفحه
        status = f"SYM:{n}  NEON:{'ON' if neon else 'OFF'}  RAINBOW:{'ON' if rainbow else 'OFF'}"
        cv2.putText(output, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

        cv2.imshow("Virtual Paint v2", output)

        # مدیریت کلیدهای کیبورد
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), np.uint8)
        elif key == ord('m'):
            sym_index = (sym_index + 1) % len(SYMMETRY_LEVELS)
        elif key == ord('n'):
            neon = not neon
        elif key == ord('r'):
            rainbow = not rainbow

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()