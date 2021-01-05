import cv2
import glob

import numpy as np


def classifier(img_path):
    for img in img_path:
        input_img = cv2.imread(img)

        img_red = input_img[:, :, 2]
        gaussian = cv2.GaussianBlur(img_red, (7, 7), 6)
        median = cv2.medianBlur(gaussian, 3)
        thresh = cv2.threshold(median, 0, 255, cv2.THRESH_OTSU)[1]

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41), (-1, -1))
        eroded = cv2.erode(thresh, element)

        opened = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, element)
        contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        total_money = 0

        for i in range(len(contours)):
            coin_type = classify_coin(contours[i], hierarchy[0][i])

            if coin_type[0] != 'Null':
                total_money += coin_type[1]
                ellipse = cv2.fitEllipse(contours[i])
                cv2.ellipse(input_img, ellipse, color=(0, 255, 0), thickness=2)
                cv2.putText(input_img, coin_type[0], (contours[i][0][0][0], contours[i][0][0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(input_img, 'Money: ' + str(np.round(total_money, 2)) + ' euro', (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        display_img('Img', input_img)


def classify_coin(contour, hierarchy):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if 4000 < area < 20000 and 230 < perimeter < 450 and (hierarchy[2] == -1 and hierarchy[3] == -1):

        if 13000 < area < 14000:
            return '50cent', 0.50
        elif 11700 < area < 12800:
            return '1euro', 1.0
        elif 10000 < area < 11200:
            return '20cent', 0.20
        elif 8800 < area < 9800:
            return '5cent', 0.05
        elif 7200 < area < 8200:
            return '10cent', 0.10
        elif 6000 < area < 7000:
            return '2cent', 0.02
        elif 4000 < area < 5000:
            return '1cent', 0.01

    return 'Null', 0


def display_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)


if __name__ == "__main__":
    path = "imagens_treino/*.jpg"

    img_path = glob.glob(path)
    img_path.sort()
    classifier(img_path)

