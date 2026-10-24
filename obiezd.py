import atexit
import time
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

# --- ОСНОВНЫЕ ПАРАМЕТРЫ ДВИЖЕНИЯ (с твоего скриншота) ---
BASE_SPEED = 1570     # базовая скорость движения по прямой линии
TURN_SPEED = 1572     # скорость в поворотах и при перестроении
ANGLE_TOLERANCE = 20  # допуск зоны прямой: |angle - 90| <= ANGLE_TOLERANCE
THRESHOLD = 205       # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0'

# --- ТАЙМИНГИ И УГЛЫ ПЕРЕСТРОЕНИЯ В ЛЕВУЮ ПОЛОСУ ---
STOP_PAUSE_TIME = 1.0    # пауза при остановке перед красным светофором (сек)
L_CHANGE_1_TIME = 1.6  # время поворота налево в 1-й фазе перестроения (сек)
L_CHANGE_2_TIME = 1.4    # время выравнивания направо во 2-й фазе перестроения (сек)
LEFT_TURN_ANGLE = 150    # угол колёс для ухода в левую полосу (на айкаре >90 — влево)
RIGHT_ALIGN_ANGLE = 60   # угол колёс для выравнивания в левой полосе (<90 — вправо)
MANEUVER_COOLDOWN = 2.0  # сколько секунд не реагировать на светофор сразу после завершения объезда

# --- НАСТРОЙКИ ДЕТЕКЦИИ КРАСНОГО СВЕТОФОРА ---
RED_LOWER1 = np.array([0, 135, 205])
RED_UPPER1 = np.array([5, 255, 255])
RED_LOWER2 = np.array([168, 135, 205])
RED_UPPER2 = np.array([180, 255, 255])

# Порог площади (в масштабе 640x480):
# Чем МЕНЬШЕ число, тем ДАЛЬШЕ от светофора начинается маневр.
# Чем БОЛЬШЕ число, тем БЛИЖЕ к светофору подъедет машинка.
RED_AREA_THRESHOLD = 800
MIN_LED_CORE_PIXELS = 12  # уменьшено с 60 до 12, чтобы не блокировать раннее обнаружение издалека
# ------------------------------------------------------------------------

# Состояния беспилотника
GO = 'GO'                  # движение по полосе с поиском светофора (на каждом круге)
STOP = 'STOP'              # пауза перед маневром (если STOP_PAUSE_TIME > 0)
L_CHANGE_1 = 'L_CHANGE_1'  # 1-я фаза перестроения (поворот налево)
L_CHANGE_2 = 'L_CHANGE_2'  # 2-я фаза перестроения (выравнивание направо)

STATE = GO
PREV_STATE = None
state_start_time = 0.0
last_maneuver_end_time = 0.0  # время окончания последнего объезда

arduino = None

@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()

arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")


def check_red_traffic_light(orig_img):
    """
    Быстрая детекция горящего красного светофора по всей ширине дороги (включая центр полосы).
    Возвращает (red_detected, max_red_area).
    """
    img_small = cv2.resize(orig_img, (640, 480))
    # Берем всю ширину перед машинкой (x: от 60 до 640) и высоту ниже стульев под потолком (y: от 100 до 420)
    roi = img_small[100:420, 60:640]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_r1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_r2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    kernel = np.ones((7, 7), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    max_red_area = 0
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_red:
        area = cv2.contourArea(c)
        if area > 80:
            x, y, w, h = cv2.boundingRect(c)
            box_bgr = roi[y:y+h, x:x+w].astype(np.int16)
            B, G, R = box_bgr[:, :, 0], box_bgr[:, :, 1], box_bgr[:, :, 2]
            # Проверка яркого центра светодиода
            led_core = np.sum((R >= 245) & (G >= 185) & ((R - B) >= 45))
            if led_core >= MIN_LED_CORE_PIXELS and area > max_red_area:
                max_red_area = area

    red_detected = max_red_area >= RED_AREA_THRESHOLD
    return red_detected, int(max_red_area)


# Настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2  # функция поиска линий разметки

# Пропускаем часть кадров для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()

    if PREV_STATE != STATE:
        print(f'>>> STATE CHANGED: {STATE}')
        PREV_STATE = STATE

    # =========================================================================
    # СОСТОЯНИЕ GO: Движение по разметке и поиск светофора (на каждом круге!)
    # =========================================================================
    if STATE == GO:
        road_frame = frame[-720:, :]
        road_frame = cv2.resize(road_frame, SIZE)
        gray = cv2.cvtColor(road_frame, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.inRange(gray, THRESHOLD, 255)

        wrapped = trans_perspective(bin_img, TRAP, RECT, SIZE)
        left, right = find_lines(wrapped)

        err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
        angle = int(90 + KP * err + KD * (err - last_err))
        last_err = err
        angle = min(max(45, angle), 135)

        # Проверяем светофор, если прошло время MANEUVER_COOLDOWN после прошлого перестроения
        red_area = 0
        if time.monotonic() - last_maneuver_end_time >= MANEUVER_COOLDOWN:
            is_red, red_area = check_red_traffic_light(orig_frame)
            if is_red:
                state_start_time = time.monotonic()
                if STOP_PAUSE_TIME > 0:
                    print(f'[TRAFFIC LIGHT] Красный (area={red_area}) -> Пауза STOP!')
                    STATE = STOP
                    arduino.set_speed(1500)
                else:
                    print(f'[TRAFFIC LIGHT] Красный (area={red_area}) -> Сразу перестроение L_CHANGE_1!')
                    STATE = L_CHANGE_1
                continue

        # Едем по полосе
        if abs(angle - 90) <= ANGLE_TOLERANCE:
            arduino.set_speed(BASE_SPEED)
        else:
            arduino.set_speed(TURN_SPEED)
        arduino.set_angle(angle)

        print(f'STATE: {STATE} | Angle: {angle} | RedArea: {red_area}')

    # =========================================================================
    # СОСТОЯНИЕ STOP: Пауза перед светофором (если STOP_PAUSE_TIME > 0)
    # =========================================================================
    elif STATE == STOP:
        arduino.set_speed(1500)
        if time.monotonic() - state_start_time >= STOP_PAUSE_TIME:
            print('[MANEUVER] Пауза завершена -> Перестроение влево (L_CHANGE_1)')
            STATE = L_CHANGE_1
            state_start_time = time.monotonic()

    # =========================================================================
    # СОСТОЯНИЕ L_CHANGE_1: Фаза 1 — уход в соседнюю полосу
    # =========================================================================
    elif STATE == L_CHANGE_1:
        arduino.set_angle(LEFT_TURN_ANGLE)
        arduino.set_speed(TURN_SPEED)
        if time.monotonic() - state_start_time >= L_CHANGE_1_TIME:
            print('[MANEUVER] Переход во 2-ю фазу — выравнивание (L_CHANGE_2)')
            STATE = L_CHANGE_2
            state_start_time = time.monotonic()

    # =========================================================================
    # СОСТОЯНИЕ L_CHANGE_2: Фаза 2 — выравнивание и возврат в состояние GO
    # =========================================================================
    elif STATE == L_CHANGE_2:
        arduino.set_angle(RIGHT_ALIGN_ANGLE)
        arduino.set_speed(TURN_SPEED)
        if time.monotonic() - state_start_time >= L_CHANGE_2_TIME:
            print('[MANEUVER] Объезд завершен -> Возвращаемся в режим GO (готовы к следующему кругу)')
            last_err = 0
            last_maneuver_end_time = time.monotonic()
            STATE = GO

    # Контроль FPS цикла
    fps = 1.0 / (time.time() - loop_start)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
