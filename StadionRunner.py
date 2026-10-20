
import atexit
import time
import cv2
import numpy as np
from pathlib import Path

from Arduino_A26 import Arduino
from road_utils import *

""" 
    Запускать на бортовом компьютере беспилотника (без монитора).
    Рядом должны быть: Arduino_A26.py, road_utils.py, 
    а также файлы нейросети: yolov4-tiny-obj.cfg, yolov4-tiny-obj_last.weights, obj1.names
"""

# ==========================================
# 1. НАСТРОЙКИ ДВИЖЕНИЯ И КАМЕРЫ
# ==========================================
BASE_SPEED = 1570 # базовая скорость движения по прямой
TURN_SPEED = 1570  # скорость в поворотах
THRESHOLD = 205  # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'

ARDUINO_PORT = '/dev/ttyUSB0'

# ==========================================
# 2. НАСТРОЙКИ НЕЙРОСЕТИ (YOLO)
# ==========================================
YOLO_CFG = "yolov4-tiny-obj.cfg"
YOLO_WEIGHTS = "yolov4-tiny-obj_last.weights"
YOLO_NAMES = "obj1.names"
CONFIDENCE_THRESHOLD = 0.5
YOLO_SKIP_FRAMES = 3  # Пропускать кадры для нейросети, чтобы не просаживать FPS

GO = 'GO'
STOP = 'STOP'

STATE = GO
PREV_STATE = None

ANGLE_TOLERANCE = 20  # допуск зоны прямой: |angle - 90| <= ANGLE_TOLERANCE

arduino = None
net = None
output_layers = []
classes = []


@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()
    # Графические окна полностью удалены, cv2.destroyAllWindows() не нужен


# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ
# ==========================================
arduino = Arduino(ARDUINO_PORT)
print("[OK] Arduino connected")

# --- Загрузка нейросети ---
print("[INFO] Loading YOLO model...")
try:
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Или DNN_TARGET_CUDA, если есть GPU

    with open(YOLO_NAMES, 'r', encoding='utf-8') as f:
        classes = [line.strip().lower() for line in f.readlines()]

    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    if isinstance(unconnected[0], (list, np.ndarray)):
        output_layers = [layer_names[i[0] - 1] for i in unconnected]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected]
    print(f"[OK] YOLO loaded. Classes: {classes}")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO: {e}")
    print("[WARNING] Continuing without neural network (STOP sign detection disabled)")

# Настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2

# Пропускаем часть кадров для стабилизации камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0
frame_count = 0

# ==========================================
# 4. ОСНОВНОЙ ЦИКЛ (Headless)
# ==========================================
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ==========================================
    # А. НЕЙРОСЕТЕВОЕ ОБНАРУЖЕНИЕ ЗНАКА СТОП
    # ==========================================
    stop_detected = False

    # Запускаем нейросеть не каждый кадр, а через YOLO_SKIP_FRAMES для экономии CPU
    if net is not None and (frame_count % YOLO_SKIP_FRAMES == 0):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    class_name = classes[class_id]
                    # Проверяем, содержит ли имя класса слово "stop"
                    if "stop" in class_name:
                        stop_detected = True
                        break  # Знак найден, можно прервать внутренний цикл

    # Логика переключения состояния
    if stop_detected:
        STATE = STOP
    else:
        STATE = GO

        # ==========================================
    # Б. ОБРАБОТКА ДОРОЖНОЙ РАЗМЕТКИ
    # ==========================================
    # Обрезаем кадр для поиска разметки (верхняя часть отбрасывается)
    crop_frame = frame[-720:, :]
    crop_frame = cv2.resize(crop_frame, SIZE)

    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.inRange(gray, THRESHOLD, 255)

    wrapped = trans_perspective(bin_img, TRAP, RECT, SIZE)
    left, right = find_lines(wrapped)

    # ПИД-регулятор
    err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
    angle = int(90 + KP * err + KD * (err - last_err))
    last_err = err
    angle = min(max(45, angle), 135)

    # ==========================================
    # В. УПРАВЛЕНИЕ ARDUINO
    # ==========================================
    if PREV_STATE != STATE:
        print(f'>>> STATE CHANGED TO: {STATE} <<<')
        PREV_STATE = STATE

    if STATE != STOP:
        # Движение
        if abs(angle - 90) <= ANGLE_TOLERANCE:
            arduino.set_speed(BASE_SPEED)
        else:
            arduino.set_speed(TURN_SPEED)
        arduino.set_angle(angle)
    else:
        # ОСТАНОВКА ПО ЗНАКУ СТОП
        arduino.set_speed(1500)  # Стоп-сигнал для моторов (нейтраль)
        arduino.set_angle(90)  # Выравниваем колеса при остановке
        # arduino.stop()         # Раскомментируй, если в твоей библиотеке есть отдельная команда стоп

    # Мониторинг производительности в консоли
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')

# Очистка при выходе из цикла
cap.release()
print("[INFO] Camera released. Exiting.")
