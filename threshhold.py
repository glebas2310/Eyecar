"""Подбор порога бинаризации белой дорожной разметки с камеры."""

import cv2
import numpy as np

CAMERA_INDEX = '/dev/video0'


def make_marking_mask(frame, threshold):
    """Возвращает одноканальную маску: белое — вероятная разметка."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {CAMERA_INDEX}")

    window = "Road marking"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window, 170, 255, lambda value: None)

    print("Двигайте ползунок Threshold. S — сохранить кадр и маску, Q — выход.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Не удалось получить кадр с камеры")
                break

            threshold = cv2.getTrackbarPos("Threshold", window)
            mask = make_marking_mask(frame, threshold)

            # Слева исходный кадр с зелёной подсветкой, справа бинарная маска.
            overlay = frame.copy()
            overlay[mask > 0] = (0.4 * frame[mask > 0] +
                                 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow(window, np.hstack((overlay, mask_bgr)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                cv2.imwrite("camera_frame.png", frame)
                cv2.imwrite("road_mask.png", mask)
                print(f"Сохранены camera_frame.png и road_mask.png; порог: {threshold}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
