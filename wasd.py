import sys
import select
import tty
import termios
import atexit
import time
from Arduino_A26 import Arduino

# Отключаем вывод "Sent to Arduino:" из библиотеки
def silent_send_data(self, data: str):
    msg = '*' + data + '|'
    self.serial.write(msg.encode(self.charset))
    self.serial.flush()

Arduino.send_data = silent_send_data

# ==========================================
# НАСТРОЙКИ ПОРТА И ДВИЖЕНИЯ
# ==========================================
ARDUINO_PORT = '/dev/ttyUSB0'

SPEED_FORWARD = 1570   # Скорость вперед (> 1500)
SPEED_BACKWARD = 1400  # Скорость назад (< 1500)
SPEED_STOP = 1500      # Остановка / нейтраль

ANGLE_LEFT = 115       # Поворот колес влево
ANGLE_RIGHT = 65      # Поворот колес вправо
ANGLE_CENTER = 90      # Колеса прямо

current_speed = SPEED_STOP
current_angle = ANGLE_CENTER

# ==========================================
# ИНИЦИАЛИЗАЦИЯ И ОЧИСТКА
# ==========================================
arduino = Arduino(ARDUINO_PORT)

@atexit.register
def cleanup():
    if arduino is not None:
        try:
            arduino.set_speed(SPEED_STOP)
            arduino.set_angle(ANGLE_CENTER)
            arduino.close()
        except Exception:
            pass
    print("\n[INFO] Соединение закрыто, машинка остановлена.")

# ==========================================
# ЧТЕНИЕ СИМВОЛОВ ИЗ ТЕРМИНАЛА (RAW MODE)
# ==========================================
def get_key(timeout=0.1):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            char = sys.stdin.read(1)
        else:
            char = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char

# ==========================================
# ИНСТРУКЦИЯ И ОСНОВНОЙ ЦИКЛ
# ==========================================
print("=" * 45)
print(" РУЧНОЕ УПРАВЛЕНИЕ EYECAR (WASD)")
print("=" * 45)
print(" [W] - Вперед")
print(" [S] - Назад")
print(" [A] - Руль влево")
print(" [D] - Руль вправо")
print(" [Пробел] - Тормоз (стоп и руль прямо)")
print(" [X] - Выровнять руль прямо")
print(" [Q] - Выход из программы")
print("=" * 45)

while True:
    key = get_key(timeout=0.05).lower()

    if key == 'w':
        current_speed = SPEED_FORWARD
        arduino.set_speed(current_speed)
        print(f"\rДвижение: ВПЕРЕД ({current_speed})   ", end='', flush=True)

    elif key == 's':
        current_speed = SPEED_BACKWARD
        arduino.set_speed(current_speed)
        print(f"\rДвижение: НАЗАД ({current_speed})    ", end='', flush=True)

    elif key == 'a':
        current_angle = ANGLE_LEFT
        arduino.set_angle(current_angle)
        print(f"\rРуль: ВЛЕВО ({current_angle})         ", end='', flush=True)

    elif key == 'd':
        current_angle = ANGLE_RIGHT
        arduino.set_angle(current_angle)
        print(f"\rРуль: ВПРАВО ({current_angle})        ", end='', flush=True)

    elif key == 'x':
        current_angle = ANGLE_CENTER
        arduino.set_angle(current_angle)
        print(f"\rРуль: ПРЯМО ({current_angle})         ", end='', flush=True)

    elif key == ' ':
        current_speed = SPEED_STOP
        current_angle = ANGLE_CENTER
        arduino.set_speed(current_speed)
        arduino.set_angle(current_angle)
        print(f"\rСТОП: Колеса прямо и мотор выключен", end='', flush=True)

    elif key == 'q':
        break

    time.sleep(0.01)
