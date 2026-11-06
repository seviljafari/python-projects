import time
import cv2
import numpy as np
import mediapipe as mp

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
BRUSH_FINGER = 20
BRUSH_PALM = 110
BLUR_KERNEL = 99
COUNTDOWN_SECONDS = 3

LIPS_PUCKER_RATIO = 0.45
LIPS_DETECT_CONFIDENCE = 0.5

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def make_fog(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    white = np.full_like(blurred, 255)
    return cv2.addWeighted(blurred, 0.45, white, 0.55, 0)


def is_open_palm(lm) -> bool:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    if not all(lm[t].y < lm[p].y for t, p in zip(tips, pips)):
        return False
    extended = 0
    for t, p in zip(tips, pips):
        if abs(lm[t].y - lm[p].y) > 0.04:
            extended += 1
    return extended >= 3


def is_lips_puckered(face_landmarks, w, h) -> bool:
    if face_landmarks is None:
        return False
    lm = face_landmarks.landmark
    lc = (lm[61].x * w, lm[61].y * h)
    rc = (lm[291].x * w, lm[291].y * h)
    tl = (lm[13].x * w, lm[13].y * h)
    bl = (lm[14].x * w, lm[14].y * h)
    mw = np.hypot(rc[0] - lc[0], rc[1] - lc[1])
    mh = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    if mw < 1:
        return False
    ratio = mh / mw
    fw = np.hypot(lm[234].x * w - lm[454].x * w, lm[234].y * h - lm[454].y * h)
    return ratio < LIPS_PUCKER_RATIO and (mw / max(fw, 1)) < 0.35


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("webcam failed")

    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                 min_detection_confidence=LIPS_DETECT_CONFIDENCE,
                                 min_tracking_confidence=0.5)

    mask = None
    prev_points = {}
    countdown_start = None
    shot_flash = 0
    puckered_frames = 0
    pucker_was_active = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)
        face_result = face_mesh.process(rgb)

        lips_puckered = False
        if face_result.multi_face_landmarks:
            lips_puckered = is_lips_puckered(face_result.multi_face_landmarks[0], w, h)

        if lips_puckered:
            puckered_frames += 1
        else:
            puckered_frames = 0
        pucker_active = puckered_frames >= 3

        if pucker_active and not pucker_was_active:
            mask[:] = 0
            prev_points.clear()
        pucker_was_active = pucker_active

        seen = set()
        if hand_result.multi_hand_landmarks:
            for idx, hand in enumerate(hand_result.multi_hand_landmarks):
                lm = hand.landmark
                seen.add(idx)

                if is_open_palm(lm):
                    cx = int(np.mean([lm[i].x for i in (0, 5, 9, 13, 17)]) * w)
                    cy = int(np.mean([lm[i].y for i in (0, 5, 9, 13, 17)]) * h)
                    pt, radius = (cx, cy), BRUSH_PALM
                else:
                    pt = (int(lm[8].x * w), int(lm[8].y * h))
                    radius = BRUSH_FINGER

                if idx in prev_points:
                    cv2.line(mask, prev_points[idx], pt, 1.0, radius * 2, cv2.LINE_AA)
                cv2.circle(mask, pt, radius, 1.0, -1, cv2.LINE_AA)
                prev_points[idx] = pt

        for k in list(prev_points):
            if k not in seen:
                del prev_points[k]

        soft = cv2.GaussianBlur(mask, (7, 7), 0)[..., None]
        fog = make_fog(frame)
        out = (frame * soft + fog * (1.0 - soft)).astype(np.uint8)

        if pucker_active:
            cv2.putText(out, "BREATHING...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining > 0:
                cv2.putText(out, str(remaining), (w // 2 - 40, h // 2 + 50),
                            cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 8, cv2.LINE_AA)
            else:
                name = time.strftime("fog_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, out)
                print("Saved:", name)
                countdown_start = None
                shot_flash = 6

        if shot_flash > 0:
            out = cv2.addWeighted(out, 0.5, np.full_like(out, 255), 0.5, 0)
            shot_flash -= 1

        cv2.imshow("Fog x Hand Tracking", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            mask[:] = 0
            prev_points.clear()
            print("Full fog")
        elif key == ord("s") and countdown_start is None:
            countdown_start = time.time()

    cap.release()
    hands.close()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()