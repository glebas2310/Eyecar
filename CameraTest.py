import cv2
import sys

def test_camera(camera_index=0):
    # Подключаемся к камере по её индексу
    print(f"Попытка подключения к камере с индексом {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    # Проверяем, удалось ли открыть камеру
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть камеру с индексом {camera_index}.")
        print("Проверьте физическое подключение камеры или попробуйте другой индекс.")
        sys.exit(1)

    print("Камера успешно подключена! Для выхода нажмите клавишу 'q'.")

    while True:
        # Захватываем кадр за кадром
        ret, frame = cap.read()

        # Если кадр не считался, выходим из цикла
        if not ret:
            print("Ошибка: Не удалось получить кадр с камеры.")
            break

        # Отображаем полученный кадр в окне
        cv2.imshow(f'Camera Test (Index {camera_index})', frame)

        # Ждем нажатия клавиши 'q' в течение 1 миллисекунды для выхода
        if cv2.getBuildInformation() and (cv2.waitKey(1) & 0xFF == ord('q')):
            print("Тестирование завершено пользователем.")
            break

    # Освобождаем ресурсы камеры и закрываем все окна
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # По умолчанию проверяем камеру с индексом 0. 
    # Если у вас несколько камер, вы можете передать другой индекс, например test_camera(1)
    test_camera(camera_index=1)
