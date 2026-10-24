import argparse
import atexit
import queue
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

# Ограничиваем потоки OpenCV, чтобы фоновая нейросеть не мешала рулению по полосе
cv2.setNumThreads(2)

# --- НАСТРОЙКИ СКОРОСТИ (СТРОГО В ДИАПАЗОНЕ 1565 - 1575 + СТОП 1500) ---
STOP_SPEED = 1500     # скорость полной остановки (на дистанции <= 30 см)
MIN_SPEED = 1565      # минимальная скорость движения (при приближении к 31-32 см)
BASE_SPEED = 1570     # средняя скорость движения (на дистанции ~36-37 см)
MAX_SPEED = 1570      # максимальная скорость движения (когда догоняем лидера на 40-45 см)

# --- НАСТРОЙКИ ДИСТАНЦИИ (ПО КООРДИНАТЕ Y В КАДРЕ ВЫСОТОЙ 720px) ---
# Чем БОЛЬШЕ y_pos, тем НИЖЕ номер в кадре (тем БЛИЖЕ к нам ведущий айкар):
Y_STOP_30CM = 50     # координата Y на расстоянии 30 см -> ПОЛНАЯ ОСТАНОВКА (1500)
Y_RESUME_MOVING = 150 # координата Y (~32-33 см), когда стоящий лидер поехал -> ТРОГАЕМСЯ
Y_TARGET_FOLLOW = 265 # целевая координата Y в движении (~36-37 см, середина коридора 30..45 см)
KP_SPEED = 0.10       # коэффициент П-регулятора для плавного хода в диапазоне 1565..1575

# --- ПАРАМЕТРЫ УДЕРЖАНИЯ ПОЛОСЫ ---
THRESHOLD = 225       # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0'

# --- ПУТИ К ТВОЕЙ НЕЙРОСЕТИ (yolov4-tiny-aicar_best.weights) ---
SEARCH_DIRS = [Path('/home/pi/base'), Path(__file__).resolve().parent]
WEIGHTS_CANDIDATES = [
    'yolov4-tiny-custom_best.weights',
    'yolov4-tiny-aicar_final.weights',
    'yolov4-tiny-aicar_last.weights',
    'yolov4-tiny-signs_best.weights',
]
CFG_CANDIDATES = [
    'yolov4-tiny-custom.cfg',
    'yolov4-tiny-signs.cfg',
]

CONF_THRESHOLD = 0.80  # порог уверенности нейросети
NMS_THRESHOLD = 0.40
YOLO_SLEEP_SEC = 0.08  # пауза потока нейросети, чтобы руление шло в 30 FPS

RECORDS_DIR = Path(__file__).resolve().parent / 'records'
RECORD_FPS = 30.0

parser = argparse.ArgumentParser(description='БПА 6: Следование за ведущим авто с остановкой на 30 см')
parser.add_argument('--name', type=str, default=None, help='Имя видеофайла')
args = parser.parse_args()
VIDEO_NAME = args.name

# На старте стоим в 30 см от ведущего и ждем, пока он поедет
CAR_SPEED = STOP_SPEED
IS_STOPPED = True
frame_cnt = 0

arduino = None
video_writer = None


@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()
    if video_writer is not None:
        video_writer.close()


arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")


def find_file(candidates, ext_fallback=None):
    """Автоматически ищет файл модели в /home/pi/s и в папке со скриптом."""
    for d in SEARCH_DIRS:
        for name in candidates:
            p = d / name
            if p.exists():
                return str(p)
    if ext_fallback:
        for d in SEARCH_DIRS:
            if d.exists():
                found = list(d.glob(f'*{ext_fallback}'))
                if found:
                    return str(found[0])
    raise FileNotFoundError(f"Не найден файл модели из списка: {candidates}")


def get_cfg_input_size(cfg_path):
    """Читает размер входа width x height из .cfg файла."""
    w, h = 416, 416
    with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('width='):
                w = int(line.split('=')[1].strip())
            elif line.startswith('height='):
                h = int(line.split('=')[1].strip())
    return w, h


class AsyncVideoWriter:
    """Фоновая запись видео без просадки FPS основного цикла."""

    def __init__(self, output_file, fps, frame_size):
        self.tasks = queue.Queue(maxsize=90)
        self.dropped = 0
        self.writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)
        if not self.writer.isOpened():
            print('[ERROR] Cannot open video writer:', output_file)
            quit()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _work(self):
        while True:
            frame = self.tasks.get()
            if frame is None:
                break
            self.writer.write(frame)

    def write(self, frame):
        try:
            self.tasks.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.tasks.put(None)
        self.worker.join(timeout=5)
        self.writer.release()
        if self.dropped:
            print(f'[INFO] При записи пропущено кадров: {self.dropped}')


def open_video_writer(frame_shape):
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    if VIDEO_NAME:
        filename = f'{VIDEO_NAME}.avi'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'run_{timestamp}.avi'
    output_file = RECORDS_DIR / filename
    h, w = frame_shape[:2]
    print(f'[INFO] Запись видео: {output_file}')
    return AsyncVideoWriter(output_file, RECORD_FPS, (w, h))


# --- ЗАГРУЗКА НЕЙРОСЕТИ ---
YOLO_WEIGHTS = find_file(WEIGHTS_CANDIDATES, ext_fallback='.weights')
YOLO_CFG = find_file(CFG_CANDIDATES, ext_fallback='.cfg')
CFG_W, CFG_H = get_cfg_input_size(YOLO_CFG)

print(f"[MODEL] Загружаем веса: {YOLO_WEIGHTS}")
print(f"[MODEL] Загружаем конфиг: {YOLO_CFG} (вход {CFG_W}x{CFG_H})")

yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = yolo_net.getLayerNames()
out_indices = yolo_net.getUnconnectedOutLayers()
if len(out_indices.shape) == 2:
    output_layers = [layer_names[i[0] - 1] for i in out_indices]
else:
    output_layers = [layer_names[i - 1] for i in out_indices]

# Переменные обмена между потоком нейросети и основным циклом
frame_lock = threading.Lock()
latest_frame_for_yolo = None
detected_y_pos = -1
detected_conf = 0.0
detected_box = None  # (x, y, w, h)
max_raw_conf = 0.0
yolo_fps = 0.0


def yolo_follower_worker():
    """
    Фоновый поток нейросети: ищет ведущий айкар и обновляет его координату y_pos.
    """
    global latest_frame_for_yolo, detected_y_pos, detected_conf, detected_box, max_raw_conf, yolo_fps

    while True:
        t0 = time.time()
        with frame_lock:
            img = latest_frame_for_yolo.copy() if latest_frame_for_yolo is not None else None

        if img is None:
            time.sleep(0.03)
            continue

        h_img, w_img = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (CFG_W, CFG_H), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        best_conf = 0.0
        best_box = None
        raw_best = 0.0

        for output in outputs:
            for det in output:
                scores = det[5:]
                conf = float(np.max(scores))
                if conf > raw_best:
                    raw_best = conf

                if conf >= CONF_THRESHOLD and conf > best_conf:
                    cx = int(det[0] * w_img)
                    cy = int(det[1] * h_img)
                    w = int(det[2] * w_img)
                    h = int(det[3] * h_img)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    best_conf = conf
                    best_box = (x, y, w, h)

        max_raw_conf = raw_best
        if best_box is not None:
            x, y, w, h = best_box
            detected_y_pos = int(y + h // 2)  # координата Y центра знака/машинки
            detected_conf = best_conf
            detected_box = best_box
        else:
            detected_y_pos = -1
            detected_conf = 0.0
            detected_box = None

        time.sleep(YOLO_SLEEP_SEC)
        dt = time.time() - t0
        yolo_fps = 1.0 / dt if dt > 0 else 0.0


# Запускаем поток нейросети
threading.Thread(target=yolo_follower_worker, daemon=True).start()


def follow_logic_30cm_stop(y_pos):
    global CAR_SPEED, IS_STOPPED, frame_cnt

    # 1. Если нейросеть в этом кадре не видит ведущую машинку
    if y_pos == -1:
        frame_cnt += 1
        # Если мы уже СТОЯЛИ в 30 см за лидером, продолжаем стоять (защита от пропуска кадров),
        # но если лидера нет в кадре очень долго (>25 кадров подряд — он уехал далеко), начинаем движение
        if IS_STOPPED:
            if frame_cnt > 25:
                IS_STOPPED = False
                CAR_SPEED = MIN_SPEED
                return f"ЛИДЕР УЕХАЛ ДАЛЕКО ({frame_cnt} кадр.) -> Старт ({CAR_SPEED})"
            CAR_SPEED = STOP_SPEED
            return f"СТОИМ НА 30 СМ (ждем лидера, не виден {frame_cnt} кадр.) -> СТОП ({CAR_SPEED})"
        else:
            # Если ехали и потеряли лидера на повороте более 10 кадров — ускоряемся до MAX_SPEED (1575)
            if frame_cnt > 10:
                CAR_SPEED = MAX_SPEED
                return f"НЕ ВИЖУ В ДВИЖЕНИИ ({frame_cnt} кадр.) -> Догоняем ({CAR_SPEED})"
            return f"НЕ ВИЖУ ({frame_cnt}/10) -> Прежняя скорость ({CAR_SPEED})"

    # Лидер виден в кадре -> сбрасываем счетчик потерянных кадров
    frame_cnt = 0

    # 2. Если лидер на дистанции 30 см или ближе -> ОСТАНАВЛИВАЕМСЯ!
    if y_pos >= Y_STOP_30CM:
        IS_STOPPED = True
        CAR_SPEED = STOP_SPEED
        return f"ДИСТАНЦИЯ <= 30 СМ (y={y_pos} >= {Y_STOP_30CM}) -> СТОИМ ({CAR_SPEED})"

    # 3. Если мы уже стояли, ждем, пока лидер реально поедет (поднимется в кадре до Y_RESUME_MOVING)
    if IS_STOPPED:
        if y_pos > Y_RESUME_MOVING:
            CAR_SPEED = STOP_SPEED
            return f"ЖДЕМ СТАРТА ЛИДЕРА (y={y_pos}, старт при y<={Y_RESUME_MOVING}) -> СТОИМ ({CAR_SPEED})"
        else:
            IS_STOPPED = False  # лидер поехал, снимаем блокировку!

    # 4. Лидер едет впереди дальше 30 см -> П-регулятор скорости в диапазоне 1565..1575
    error = Y_TARGET_FOLLOW - y_pos
    speed = int(round(BASE_SPEED + KP_SPEED * error))
    CAR_SPEED = max(MIN_SPEED, min(MAX_SPEED, speed))
    return f"ЕДЕМ ЗА ЛИДЕРОМ! y={y_pos} (err={error:+d}) -> Скорость {CAR_SPEED}"


# Настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2

for i in range(30):
    ret, frame = cap.read()

last_err = 0

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Берем нижние 720 строк кадра (как в твоем базовом коде удержания полосы)
    road_frame = frame[-720:, :]
    orig_frame = road_frame.copy()

    # Отправляем свежий кадр в поток нейросети
    with frame_lock:
        latest_frame_for_yolo = orig_frame

    # Считываем актуальные данные от нейросети
    y_pos = detected_y_pos
    conf = detected_conf
    box = detected_box
    raw_c = max_raw_conf

    # 1. Вычисляем скорость (1500 при <=30 см или 1565..1575 при движении)
    status_str = follow_logic_30cm_stop(y_pos)

    # Отрисовка рамки лидера на видеозаписи
    if box is not None:
        bx, by, bw, bh = box
        color = (0, 0, 255) if IS_STOPPED else (0, 255, 0)
        cv2.rectangle(orig_frame, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(orig_frame, f"y={y_pos} spd={CAR_SPEED}",
                    (bx, max(25, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if video_writer is None:
        video_writer = open_video_writer(orig_frame.shape)
    video_writer.write(orig_frame)

    # 2. Удержание полосы (работает на полных ~30 FPS)
    frame_resized = cv2.resize(road_frame, SIZE)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.inRange(gray, THRESHOLD, 255)

    wrapped = trans_perspective(bin_img, TRAP, RECT, SIZE)
    left, right = find_lines(wrapped)

    err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
    angle = int(90 + KP * err + KD * (err - last_err))
    last_err = err
    angle = min(max(45, angle), 135)

    # Отправляем команды на микроконтроллер
    arduino.set_speed(CAR_SPEED)
    arduino.set_angle(angle)

    main_fps = 1.0 / (time.time() - loop_start)

    # Вывод в терминал на каждом кадре
    print(
        f"FPS:{main_fps:4.1f} (YOLO:{yolo_fps:3.1f}) | "
        f"Руль:{angle:3d} | RawConf:{raw_c:.2f} | {status_str}"
    )
