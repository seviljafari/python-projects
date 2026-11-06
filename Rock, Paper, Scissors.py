import cv2
import mediapipe as mp
import random
import time
import numpy as np
import threading
import winsound
from collections import deque, Counter

# ============ تنظیمات ============
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
WINDOW_W, WINDOW_H = 700, 500
WIN_SCORE = 10
COUNTDOWN_TIME = 1.5

# ============ MediaPipe ============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ============ دوربین ============
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
if not cap.isOpened():
    raise RuntimeError("Webcam not opened.")

# ============ وضعیت بازی ============
scores = [0, 0]
state = "waiting"
countdown_start = 0
player_move = ""
computer_move = ""
message = ""
result_color = (255, 255, 255)
effect_start = 0
particles = []
pulse = 0
last_countdown_num = -1

# ============ AI: حافظه ============
history = deque(maxlen=20)
transition = {"ROCK": Counter(), "PAPER": Counter(), "SCISSORS": Counter()}
last_two = deque(maxlen=2)
BEAT = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}

# ============ رنگ‌ها ============
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
NEON_GREEN = (0, 255, 128)
NEON_RED = (80, 80, 255)
WHITE = (255, 255, 255)
GRAY = (140, 140, 160)
DARK_PANEL = (25, 20, 40)


# ============ صدا ============
def play_notes(notes, duration=100):
    """پخش نُت‌ها در thread جداگانه"""
    def _play():
        try:
            for n in notes:
                winsound.Beep(n, duration)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


def sound_win():
    play_notes([523, 659, 784, 1047], 100)

def sound_lose():
    play_notes([392, 330, 262, 196], 130)

def sound_draw():
    play_notes([440, 440], 150)

def sound_tick():
    play_notes([880], 60)

def sound_go():
    play_notes([1047, 1319], 80)

def sound_firework():
    play_notes([1319, 1568, 1760, 2093], 50)


# ============ توابع بازی ============
def detect_move(landmarks):
    """تشخیص ROCK / PAPER / SCISSORS از تعداد انگشتان"""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers_up = 0
    for t, p in zip(tips, pips):
        if landmarks[t].y < landmarks[p].y:
            fingers_up += 1
    if fingers_up == 0:
        return "ROCK"
    elif fingers_up == 2:
        return "SCISSORS"
    elif fingers_up == 4:
        return "PAPER"
    return None


def winner(p, c):
    """0=مساوی، 1=تو، 2=کامپیوتر"""
    if p == c:
        return 0
    wins = {"ROCK": "SCISSORS", "PAPER": "ROCK", "SCISSORS": "PAPER"}
    if wins.get(p) == c:
        return 1
    return 2


def smart_ai():
    """AI: الگوی ۲ حرکتی → الگوی ۱ حرکتی → تصادفی"""
    if len(last_two) == 2:
        key = tuple(last_two)
        matches = []
        for i in range(len(history) - 1):
            if i >= 1 and (history[i-1], history[i]) == key:
                if i + 1 < len(history):
                    matches.append(history[i+1])
        if matches:
            predicted = Counter(matches).most_common(1)[0][0]
            return BEAT[predicted]
    if len(history) >= 3:
        last = history[-1]
        if sum(transition[last].values()) >= 2:
            predicted = transition[last].most_common(1)[0][0]
            if transition[last][predicted] >= 2:
                return BEAT[predicted]
    return random.choice(["ROCK", "PAPER", "SCISSORS"])


def record_move(move):
    """ثبت حرکت برای یادگیری AI"""
    if history:
        transition[history[-1]][move] += 1
    history.append(move)
    last_two.append(move)


# ============ رسم ============
def text_center(frame, text, y, color, scale=1.2, thick=3):
    """متن وسط صفحه با سایه"""
    h, w = frame.shape[:2]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0]
    x = (w - size[0]) // 2
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def gradient_bg(frame, top_color, bottom_color):
    """پس‌زمینه گرادیانت"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for y in range(h):
        t = y / h
        b = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        r = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        cv2.line(overlay, (0, y), (w, y), (b, g, r), 1)
    frame[:] = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


def glow_rect(frame, x1, y1, x2, y2, color, thickness=2, radius=20):
    """کادر نئون با گوشه‌گرد"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame[:] = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def progress_bar(frame, x1, y1, x2, y2, progress, color):
    """نوار پیشرفت امتیاز"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 60), -1)
    w = int((x2 - x1) * progress)
    if w > 0:
        cv2.rectangle(frame, (x1, y1), (x1 + w, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def glass_card(frame, x, y, w_card, h_card, text, color):
    """کارت شیشه‌ای برای نمایش حرکت"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_card, y + h_card), color, -1)
    frame[:] = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    cv2.rectangle(frame, (x, y), (x + w_card, y + h_card), color, 2)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
    tx = x + (w_card - size[0]) // 2
    ty = y + (h_card + size[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA)


def spawn_particles(cx, cy, color, count=60):
    """انفجار فشفشه"""
    for _ in range(count):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2, 8)
        particles.append({
            "x": cx, "y": cy,
            "vx": np.cos(angle) * speed,
            "vy": np.sin(angle) * speed,
            "life": 1.0,
            "color": color,
            "size": random.randint(2, 5),
        })


def update_particles(frame):
    """حرکت و محو ذرات"""
    global particles
    new_particles = []
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["vy"] += 0.3
        p["life"] -= 0.03
        if p["life"] > 0:
            alpha = max(0, p["life"])
            color = tuple(int(c * alpha) for c in p["color"])
            cv2.circle(frame, (int(p["x"]), int(p["y"])), p["size"], color, -1)
            new_particles.append(p)
    particles = new_particles


# ============ پنجره ============
window = "RPS - Reels Edition"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, WINDOW_W, WINDOW_H)

# ============ حلقه اصلی ============
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    pulse = (pulse + 1) % 60

    # تشخیص دست و رسم اسکلت
    detected = None
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        detected = detect_move(lm)
        for a, b in mp_hands.HAND_CONNECTIONS:
            p1 = (int(lm[a].x * w), int(lm[a].y * h))
            p2 = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(frame, p1, p2, MAGENTA, 2, cv2.LINE_AA)
        for i in range(21):
            cx, cy = int(lm[i].x * w), int(lm[i].y * h)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, CYAN, -1, cv2.LINE_AA)

    gradient_bg(frame, (40, 20, 60), (10, 5, 25))

    # HUD بالا
    cv2.rectangle(frame, (0, 0), (w, 110), DARK_PANEL, -1)
    cv2.putText(frame, "YOU", (30, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, NEON_GREEN, 2)
    cv2.putText(frame, str(scores[0]), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.8, WHITE, 3)
    cv2.putText(frame, "AI", (w - 90, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, NEON_RED, 2)
    cv2.putText(frame, str(scores[1]), (w - 90, 80), cv2.FONT_HERSHEY_DUPLEX, 1.8, WHITE, 3)
    progress_bar(frame, 130, 45, w // 2 - 30, 60, scores[0] / WIN_SCORE, NEON_GREEN)
    progress_bar(frame, w // 2 + 30, 45, w - 140, 60, scores[1] / WIN_SCORE, NEON_RED)
    text_center(frame, "VS", 55, CYAN, 0.9, 3)

    # ============ حالت انتظار ============
    if state == "waiting":
        glow_rect(frame, w // 2 - 320, h // 2 - 80, w // 2 + 320, h // 2 + 50, CYAN, 2)
        text_center(frame, "SHOW YOUR HAND", h // 2 - 15, WHITE, 1.5, 3)
        text_center(frame, "PRESS SPACE", h // 2 + 30, CYAN, 0.9, 2)
        if detected:
            cv2.putText(frame, f"Detected: {detected}", (30, h - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, NEON_GREEN, 2)

    # ============ شمارش معکوس ============
    elif state == "countdown":
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN_TIME - elapsed

        if remaining > 0:
            num = int(remaining * 2) + 1
            pulse_scale = 4 + int(pulse / 8) % 2
            text_center(frame, str(num), h // 2 + 50, NEON_RED, pulse_scale, 10)
            if num != last_countdown_num:
                sound_tick()
                last_countdown_num = num
        else:
            text_center(frame, "GO!", h // 2 + 50, NEON_GREEN, 4, 8)

        glass_card(frame, 40, h - 140, 280, 90, f"YOU: {detected if detected else '???'}", NEON_GREEN)
        glass_card(frame, w - 320, h - 140, 280, 90, f"AI: {computer_move if computer_move else '???'}", NEON_RED)

        if remaining < -0.05:
            computer_move = smart_ai()
            player_move = detected if detected else "UNKNOWN"

            if player_move == "UNKNOWN":
                message = "AI WINS!"
                scores[1] += 1
                result_color = NEON_RED
                spawn_particles(w // 2, h // 2, NEON_RED, 80)
                sound_lose()
                sound_firework()
            else:
                record_move(player_move)
                res = winner(player_move, computer_move)
                if res == 0:
                    message = "DRAW"
                    result_color = CYAN
                    spawn_particles(w // 2, h // 2, CYAN, 40)
                    sound_draw()
                elif res == 1:
                    message = "YOU WIN!"
                    scores[0] += 1
                    result_color = NEON_GREEN
                    spawn_particles(w // 2, h // 2, NEON_GREEN, 80)
                    sound_win()
                    sound_firework()
                else:
                    message = "YOU LOSE"
                    scores[1] += 1
                    result_color = NEON_RED
                    spawn_particles(w // 2, h // 2, NEON_RED, 80)
                    sound_lose()
                    sound_firework()

            effect_start = time.time()
            if scores[0] >= WIN_SCORE or scores[1] >= WIN_SCORE:
                state = "gameover"
            else:
                state = "result"

    # ============ نمایش نتیجه ============
    elif state == "result":
        elapsed_effect = time.time() - effect_start

        if message == "YOU WIN!":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), NEON_GREEN, -1)
            a = 0.15 * max(0, 1 - elapsed_effect)
            frame[:] = cv2.addWeighted(overlay, a, frame, 1 - a, 0)
        elif message == "YOU LOSE" or message == "AI WINS!":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), NEON_RED, -1)
            a = 0.15 * max(0, 1 - elapsed_effect)
            frame[:] = cv2.addWeighted(overlay, a, frame, 1 - a, 0)

        update_particles(frame)

        glow_rect(frame, w // 2 - 340, h // 2 - 120, w // 2 + 340, h // 2 + 80, result_color, 3)
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 2.2, 6)[0]
        tx = (w - size[0]) // 2
        cv2.putText(frame, message, (tx + 3, h // 2 - 30), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(frame, message, (tx, h // 2 - 35), cv2.FONT_HERSHEY_DUPLEX, 2.2, result_color, 6, cv2.LINE_AA)

        text_center(frame, f"YOU: {player_move}   |   AI: {computer_move}",
                    h // 2 + 20, WHITE, 1.0, 2)
        text_center(frame, "SPACE = next", h - 50, CYAN, 0.8, 2)

    # ============ پایان بازی ============
    elif state == "gameover":
        update_particles(frame)

        if scores[0] >= WIN_SCORE:
            glow_rect(frame, w // 2 - 360, h // 2 - 130, w // 2 + 360, h // 2 + 80, NEON_GREEN, 3)
            text_center(frame, "YOU WIN!", h // 2 - 40, NEON_GREEN, 2.4, 6)
        else:
            glow_rect(frame, w // 2 - 360, h // 2 - 130, w // 2 + 360, h // 2 + 80, NEON_RED, 3)
            text_center(frame, "AI WINS!", h // 2 - 40, NEON_RED, 2.4, 6)

        text_center(frame, f"FINAL: {scores[0]} - {scores[1]}",
                    h // 2 + 30, WHITE, 1.2, 2)
        text_center(frame, "R = restart   Q = quit", h - 50, CYAN, 0.85, 2)

    # نوار پایین
    cv2.rectangle(frame, (0, h - 30), (w, h), DARK_PANEL, -1)
    cv2.putText(frame, "Q = quit   |   SPACE = play", (20, h - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, GRAY, 1)

    cv2.imshow(window, frame)

    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        break

    if key == ord(" "):
        if state == "waiting":
            state = "countdown"
            countdown_start = time.time()
            player_move = ""
            computer_move = ""
            message = ""
            last_countdown_num = -1
            sound_go()
        elif state == "result":
            state = "waiting"

    if key == ord("r") and state == "gameover":
        scores = [0, 0]
        state = "waiting"
        message = ""
        particles = []

cap.release()
hands.close()
cv2.destroyAllWindows()
print(f"\nFINAL: YOU {scores[0]} - AI {scores[1]}")


