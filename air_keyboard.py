import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ================== HandLandmarker ==================
model_path = "models/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)
# ================== Keyboard Layout ==================
LETTERS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["Space", "Backspace", "Symbols"]
]

SYMBOLS = [
    list("!@#$%^&*()"),
    list("-_=+[]{}"),
    list(";:'\",.<>?/"),
    ["Space", "Backspace", "Letters"]
]

typed_text = ""
show_symbols = False

# ================== Pinch Logic ==================
pinch_active = False
last_press_time = 0
cooldown = 0.4
PINCH_ON = 0.28
PINCH_OFF = 0.42

# ================== Cursor ==================
CURSOR_BLINK_INTERVAL = 0.5
last_cursor_toggle = time.time()
cursor_visible = True

# ================== Open Palm Delete ==================
OPEN_PALM_TIME = 1.2
palm_start_time = None
last_palm_delete = 0
PALM_DELETE_COOLDOWN = 1.0
palm_start_time = None
last_palm_delete = 0

OPEN_PALM_TIME = 0.8          # seconds to HOLD palm
PALM_DELETE_COOLDOWN = 0.8    # seconds between deletes


# ================== Glass Panel ==================
def draw_glass_panel(frame, x1, y1, x2, y2, alpha=0.35):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

# ================== Open Palm Detection ==================
def is_open_palm(hand, w, h):
    fingers = [
        (8, 6),    # index
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18)   # pinky 
    ]

    extended = 0
    for tip, pip in fingers:
        if hand[tip].y < hand[pip].y:
            extended += 1

    if hand[4].x < hand[3].x:
        extended += 1

    return extended >= 4

# ================== Draw Keyboard ==================
def draw_keyboard(frame, hover_key=None):
    h, w, _ = frame.shape
    key_positions = {}

    keys = SYMBOLS if show_symbols else LETTERS
    key_w = w // 14
    key_h = h // 12
    gap = 10

    total_height = len(keys) * (key_h + gap)
    start_y = (h - total_height) // 2  
    y = start_y

    for row in keys:
        row_width = sum(
            (key_w * 4 if key == "Space" else key_w * 2 if key in ["Backspace", "Symbols", "Letters"] else key_w) + gap
            for key in row
        )
        x = (w - row_width) // 2

        for key in row:
            width = key_w
            if key == "Space":
                width = key_w * 4
            elif key in ["Backspace", "Symbols", "Letters"]:
                width = key_w * 2

            border = (0, 255, 255) if hover_key == key else (200, 200, 200)
            draw_glass_panel(frame, x, y, x + width, y + key_h, 0.4)
            cv2.rectangle(frame, (x, y), (x + width, y + key_h), border, 2)

            cv2.putText(
                frame, key,
                (x + 10, y + key_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            key_positions[key] = (x, y, x + width, y + key_h)
            x += width + gap
        y += key_h + gap

    return key_positions

# ================== Camera ==================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Air Keyboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Keyboard", 1280, 720)

# ================== Main Loop ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ---------- Cursor ----------
    if time.time() - last_cursor_toggle > CURSOR_BLINK_INTERVAL:
        cursor_visible = not cursor_visible
        last_cursor_toggle = time.time()

    display_text = typed_text[-50:]
    if cursor_visible:
        display_text += "|"

    # ---------- Text Panel ----------
    draw_glass_panel(frame, 40, 30, w - 40, 120)
    cv2.putText(frame, display_text, (60, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.putText(frame,
                "Pinch = Type | Open Palm (hold) = Delete",
                (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

    # ---------- Hand Detection ----------
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    result = detector.detect(mp_image)

    hovered_key = None

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        ix, iy = int(hand[8].x * w), int(hand[8].y * h)
        tx, ty = int(hand[4].x * w), int(hand[4].y * h)
        hx1, hy1 = int(hand[5].x * w), int(hand[5].y * h)
        hx2, hy2 = int(hand[17].x * w), int(hand[17].y * h)

        hand_width = np.hypot(hx2 - hx1, hy2 - hy1)
        pinch_ratio = np.hypot(ix - tx, iy - ty) / hand_width

        cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)

        key_positions = draw_keyboard(frame)

        for key, (x1, y1, x2, y2) in key_positions.items():
            if x1 < ix < x2 and y1 < iy < y2:
                hovered_key = key
                break

        now = time.time()

        # ---------- Pinch Typing ----------
        if pinch_ratio < PINCH_ON and not pinch_active and now - last_press_time > cooldown:
            pinch_active = True
            last_press_time = now

            if hovered_key:
                if hovered_key == "Space":
                    typed_text += " "
                elif hovered_key == "Backspace":
                    typed_text = typed_text[:-1]
                elif hovered_key == "Symbols":
                    show_symbols = True
                elif hovered_key == "Letters":
                    show_symbols = False
                else:
                    typed_text += hovered_key

        if pinch_ratio > PINCH_OFF:
            pinch_active = False

        # ---------- Open Palm Delete ----------
        if is_open_palm(hand, w, h) and not pinch_active:
            if palm_start_time is None:
                palm_start_time = now
            elif now - palm_start_time > OPEN_PALM_TIME and now - last_palm_delete > PALM_DELETE_COOLDOWN:
                typed_text = typed_text[:-1]
                last_palm_delete = now
                palm_start_time = None
        else:
            palm_start_time = None

    else:
        draw_keyboard(frame)

    cv2.putText(frame, "Press Q to Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Air Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()