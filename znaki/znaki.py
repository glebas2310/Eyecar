import atexit
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

# Ограничиваем потоки OpenCV для стабильного FPS
cv2.setNumThreads(2)

# --- ПАРАМЕТРЫ СКОРОСТИ ---
BASE_SPEED = 1567        # базовая скорость по прямой
TURN_SPEED = 1572        # скорость в поворотах
# Заметная для судей скорость замедления (1550-1554 - ползком, 1500 - стоп):
SLOW_SPEED = 1500        
SLOW_DURATION = 5.0      # сколько секунд ехать медленно (сек)
SLOW_COOLDOWN = 4.0      # пауза после окончания замедления до следующего знака (сек)

ANGLE_TOLERANCE = 20     # допуск зоны прямой: |angle - 90| <= ANGLE_TOLERANCE
THRESHOLD = 225          # порог бинаризации для линий
CAMERA_ID = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0'

# --- ФАЙЛЫ НЕЙРОСЕТИ (прямо в папке со скриптом) ---
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = str(SCRIPT_DIR / "yolov4-tiny-custom_best_1_weights.weights")
CONFIG_PATH = str(SCRIPT_DIR / "yolov4-tiny-custom (1).cfg")
NAMES_PATH = str(SCRIPT_DIR / "obj (1).names")

TARGET_SIGN = "peshehodniy perehod"  # класс знака в файле obj(1).names
CONF_THRESHOLD = 0.80       # порог детекции
NMS_THRESHOLD = 0.40

arduino = None
running = True

@atexit.register
def exit_func(*args):
    if arduino is not None:
        try:
            arduino.set_speed(1500)
            arduino.close()
        except Exception:
            pass

arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")

with open(NAMES_PATH, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]

net = cv2.dnn.readNet(model=WEIGHTS_PATH, config=CONFIG_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
print("[INFO] Нейросеть готова к работе!")

# Переменные обмена данными
frame_lock = threading.Lock()
latest_full_frame = None
current_detections = []
pedestrian_detected = False
best_raw_debug = "wait..."
yolo_fps = 0.0


def yolo_worker():
    """Фоновый поток для детекции знаков."""
    global latest_full_frame, current_detections, pedestrian_detected, best_raw_debug, yolo_fps

    while running:
        t0 = time.time()
        with frame_lock:
            frame_copy = latest_full_frame.copy() if latest_full_frame is not None else None

        if frame_copy is None:
            time.sleep(0.02)
            continue

        h, w = frame_copy.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_copy, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        max_raw_conf = 0.0
        max_raw_name = "none"

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])

                if conf > max_raw_conf:
                    max_raw_conf = conf
                    max_raw_name = classes[class_id] if class_id < len(classes) else f"ID {class_id}"

                if conf >= CONF_THRESHOLD:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    boxes.append([int(cx - bw / 2), int(cy - bh / 2), bw, bh])
                    confidences.append(conf)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        det_list = []
        is_pedestrian_now = False

        if len(indices) > 0:
            for i in indices.flatten():
                cid = class_ids[i]
                cname = classes[cid] if cid < len(classes) else f"ID {cid}"
                cconf = confidences[i]
                det_list.append((cname, cconf))
                if cname == TARGET_SIGN:
                    is_pedestrian_now = True

        best_raw_debug = f"{max_raw_name}:{max_raw_conf:.2f}"
        current_detections = det_list
        pedestrian_detected = is_pedestrian_now

        yolo_fps = 1.0 / max(0.001, time.time() - t0)
        time.sleep(0.01)


# Запуск камеры
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2

# Прогрев камеры
for _ in range(30):
    ret, frame = cap.read()

threading.Thread(target=yolo_worker, daemon=True).start()

last_err = 0
slow_end_time = 0.0
cooldown_end_time = 0.0

try:
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        with frame_lock:
            latest_full_frame = frame

        now = time.monotonic()

        # Проверка срабатывания на пешеходный переход
        if pedestrian_detected:
            if now > cooldown_end_time:
                # Взводим таймер замедления на 5 секунд
                slow_end_time = now + SLOW_DURATION
                cooldown_end_time = slow_end_time + SLOW_COOLDOWN
                print(f"\n=======================================================")
                print(f">>> [ЗНАК: {TARGET_SIGN}] СБРОС СКОРОСТИ ДО {SLOW_SPEED} НА {SLOW_DURATION} СЕК! <<<")
                print(f"=======================================================\n")

        # Находимся ли сейчас в режиме замедления
        in_slow_mode = (now < slow_end_time)

        # Руление по полосе (нижние 720 строк)
        road_crop = frame[-720:, :]
        road_resized = cv2.resize(road_crop, SIZE)
        gray = cv2.cvtColor(road_resized, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.inRange(gray, THRESHOLD, 255)

        wrapped = trans_perspective(bin_img, TRAP, RECT, SIZE)
        left, right = find_lines(wrapped)

        err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
        angle = int(90 + KP * err + KD * (err - last_err))
        last_err = err
        angle = min(max(45, angle), 135)

        # Определение и отправка скорости
        if in_slow_mode:
            current_speed = SLOW_SPEED
            time_left = slow_end_time - now
            mode_text = f"SLOW ({time_left:.1f}s)"
        else:
            if abs(angle - 90) <= ANGLE_TOLERANCE:
                current_speed = BASE_SPEED
            else:
                current_speed = TURN_SPEED
            mode_text = "NORMAL"

        arduino.set_speed(current_speed)
        arduino.set_angle(angle)

        fps = 1.0 / max(0.001, time.time() - t_start)
        dets = list(current_detections)
        signs_view = [f"{n}:{c:.2f}" for n, c in dets]

        if in_slow_mode:
            print(f"FPS:{fps:4.1f} | YOLO:{yolo_fps:3.1f} | >>> {mode_text} Spd:{current_speed} <<< | Ang:{angle:3d} | Знаки:{signs_view}")
        else:
            print(f"FPS:{fps:4.1f} | YOLO:{yolo_fps:3.1f} | {mode_text} Spd:{current_speed} | Ang:{angle:3d} | Знаки:{signs_view}")

except KeyboardInterrupt:
    print("\nОстановка...")
finally:
    running = False
    arduino.set_speed(1500)
