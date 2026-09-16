import atexit
import time
import random
import threading
from pathlib import Path

import cv2
import numpy as np

from Arduino import Arduino
from Utils import *

""" Запускать на бортовом компьютере беспилотника.
    Рядом должны быть Arduino.py и Utils.py.

    Беспилотник движется по дорожной разметке и использует нейросетевой детектор для
    обнаружения: пешеходов, знаков, светофоров.
    При появлении в кадре пешехода беспилотный автомобиль останавливается.

    В основном цикле организованы:
    поиск линий дорожной разметки,
    определение угла поворота колёс для движения к центру полосы,
    опрос микроконтроллера, для отслеживания дистанции, которую осталось проехать,
    обработка кадра нейросетевым детектором,
    анализ результатов работы детектора.

"""

# Меняем только эти параметры
    
BASE_SPEED = 1570    # базовая скорость движения по прямой линии
TURN_SPEED = 1575          # скорость в поворотах, когда колёса вывернуты за пределы зоны прямой
THRESHOLD = 250  # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'

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


#def speed_input():
   
#    global BASE_SPEED, TURN_SPEED
#    while True:
#        s = input('Скорость: число — прямая, "2 число" — поворот: ').strip()
#        if not s:
#            continue
#        parts = s.split()
#        try:
#            if len(parts) == 1:  # введено только число - меняем скорость по прямой
#                BASE_SPEED = int(parts[0])
#                print(f'BASE_SPEED = {BASE_SPEED}')
#            elif len(parts) == 2 and parts[0] == '2':  # префикс "2" - скорость в поворотах
#                TURN_SPEED = int(parts[1])
#                print(f'TURN_SPEED = {TURN_SPEED}')
#            else:
#                print('Неверный формат! Примеры: "1560" или "2 1530"')
#        except ValueError:
#            print('Введите целое число!')


#threading.Thread(target=speed_input, daemon=True).start()




# астраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2 # название функции для поиска линий разметки

# пропускаем часть кадров, для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Сохраняем оригинально изображение для дальнейшей обработки

    orig_frame = frame.copy()


    
    frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
    
    frame = cv2.resize(frame, SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в чёрно-белое с градациями серого
    bin = cv2.inRange(gray, THRESHOLD, 255)  # Бинаризуем по порогу, должны остаться только белые линии разметки

    wrapped = trans_perspective(bin, TRAP, RECT, SIZE)  # получаем область перед колёсами
    left, right = find_lines(wrapped)  # координаты левой и правой линий разметки

    # ПИД-регулятор для определения угла поворота колёс
    # ПИД старается удерживать центр кадра ровно между линиями дорожной разметки
    err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)

    # err = -err  # Инвертирование направления поворота колёс  На наших айкарах не надо

    angle = int(90 + KP * err + KD * (err - last_err))  # серва на роботе зеркальная: поправка входит с обратным знаком
    last_err = err

    angle = min(max(45, angle), 135)
    print(angle)


    

    if PREV_STATE != STATE:
        print(f'STATE: {STATE}')
        PREV_STATE = STATE

    if STATE != STOP:
        # Выбор скорости: в зоне прямой (90 ± ANGLE_TOLERANCE) едем на базовой скорости,
        # при вывороте колёс за пределы зоны - замедляемся для прохождения поворота
        if abs(angle - 90) <= ANGLE_TOLERANCE:
            arduino.set_speed(BASE_SPEED)
        else:
            arduino.set_speed(TURN_SPEED)
        arduino.set_angle(angle)
    else:
        arduino.set_speed(1500)  # Стоп-сигнал
        # arduino.stop()  # Ардуино подтвердит получение

    end_time = time.time()

    fps = 1 / (end_time - start_time)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
