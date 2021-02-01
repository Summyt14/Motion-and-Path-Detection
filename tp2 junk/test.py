from cv2 import cv2
import numpy as np

cap = cv2.VideoCapture('camera1.mp4')

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)
    median = cv2.medianBlur(mask, 255 // 50)
    _, thresh = cv2.threshold(median, 200, 255, cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4), (-1, -1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (-1, -1))
    dilation = cv2.dilate(thresh, kernel1, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            print(area)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', closing)

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
