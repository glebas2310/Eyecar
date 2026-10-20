import atexit
import time
import random
import threading
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

# Меняем только эти параметры
BASE_SPEED = 1565     # базовая скорость движения по прямой линии
TURN_SPEED = 1565     # скорость в поворотах
THRESHOLD = 205    # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'

# --- ТОЧНЫЕ НАСТРОЙКИ СВЕТОФОРА ПО ДВУМ ФОТО (HSV) ---
# Горящий красный свет (высокая яркость V>=205, чтобы не срабатывать на выключенный красный и оранжевые стулья)
RED_LOWER1 = np.array([0, 135, 205])
RED_UPPER1 = np.array([4, 255, 255])
RED_LOWER2 = np.array([168, 135, 205])
RED_UPPER2 = np.array([180, 255, 255])

# Горящий зеленый (бирюзовый) свет в нижней секции светофора
GREEN_LOWER = np.array([55, 90, 165])
GREEN_UPPER = np.array([108, 255, 255])

# Пороги площади (в масштабе 640x480):
# Чем БОЛЬШЕ RED_AREA_THRESHOLD, тем БЛИЖЕ к светофору подъедет машинка перед остановкой.
# На твоем фото вблизи площадь красного ~2800 пикселей.
RED_AREA_THRESHOLD = 1900   # Остановка вблизи светофора (если нужно еще ближе — поставь 2200-2400)
GREEN_AREA_THRESHOLD = 500  # Быстрый старт при появлении зеленого сигнала
#------------------------------------------------------------------------

ARDUINO_PORT = '/dev/ttyUSB0'

GO = 'GO'
STOP = 'STOP'

STATE = GO
PREV_STATE = None
PREV_SUBSTATE = None
SUBSTATE = None

ANGLE_TOLERANCE = 20  # допуск зоны прямой: |angle - 90| <= ANGLE_TOLERANCE

arduino = None
video_orig = None

@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()
    if video_orig is not None:
        video_orig.close()
    # cv2.destroyAllWindows()

arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")


def check_traffic_light(orig_img):
    """
    Приводит кадр к 640x480 и ищет красный/зеленый сигналы светофора в правой части кадра.
    Возвращает (red_detected, green_detected, red_area, green_area).
    """
    img_small = cv2.resize(orig_img, (640, 480))
    
    # Область интереса (ROI): правая половина кадра (x: 280..640, y: 60..420), где стоит светофор
    roi = img_small[60:420, 280:640]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Маска для красного сигнала
    mask_r1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask_r2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # Маска для зеленого (бирюзового) сигнала
    mask_green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Морфология (закрывает белое пятно засвета в центре диода в единый контур)
    kernel = np.ones((7, 7), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Ищем максимальную площадь контура красного цвета
    max_red_area = 0
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_red:
        max_red_area = max(cv2.contourArea(c) for c in contours_red)

    # Ищем максимальную площадь контура зеленого цвета
    max_green_area = 0
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_green:
        max_green_area = max(cv2.contourArea(c) for c in contours_green)

    red_detected = max_red_area > RED_AREA_THRESHOLD
    green_detected = max_green_area > GREEN_AREA_THRESHOLD

    return red_detected, green_detected, int(max_red_area), int(max_green_area)


# Настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2  # название функции для поиска линий разметки

# Пропускаем часть кадров для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Сохраняем оригинальное изображение для обработки светофора
    orig_frame = frame.copy()

    # --- ЛОГИКА СВЕТОФОРА ---
    is_red, is_green, red_area, green_area = check_traffic_light(orig_frame)

    if STATE == GO and is_red:
        STATE = STOP
        print(f'[TRAFFIC LIGHT] КРАСНЫЙ (area={red_area}) -> СТОП!')
    elif STATE == STOP and is_green:
        STATE = GO
        print(f'[TRAFFIC LIGHT] ЗЕЛЕНЫЙ (area={green_area}) -> ЕДЕМ!')
    # ------------------------

    frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
    
    frame = cv2.resize(frame, SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin = cv2.inRange(gray, THRESHOLD, 255)

    wrapped = trans_perspective(bin, TRAP, RECT, SIZE)
    left, right = find_lines(wrapped)

    # ПИД-регулятор для определения угла поворота колёс
    err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
    angle = int(90 + KP * err + KD * (err - last_err))
    last_err = err

    angle = min(max(45, angle), 135)
    print(f'Angle: {angle} | STATE: {STATE} | RedArea: {red_area} | GreenArea: {green_area}')

    if PREV_STATE != STATE:
        print(f'STATE changed to: {STATE}')
        PREV_STATE = STATE

    if STATE != STOP:
        if abs(angle - 90) <= ANGLE_TOLERANCE:
            arduino.set_speed(BASE_SPEED)
        else:
            arduino.set_speed(TURN_SPEED)
        arduino.set_angle(angle)
    else:
        arduino.set_speed(1500)  # Стоп-сигнал (стоим до зеленого)

    end_time = time.time()

    fps = 1 / (end_time - start_time)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
