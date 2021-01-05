import cv2
import numpy as np
import matplotlib.pyplot as plt

fileDir = 'imageDatabase/'
fileName = 'JonquilFlowers.jpg'
# fileName = 'cameraman.jpg'

img = cv2.imread(fileDir + fileName)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', imgg)

imgDx = cv2.Sobel(imgg, cv2.CV_16S, dx=1, dy=0, ksize=3)
imgDy = cv2.Sobel(imgg, cv2.CV_16S, dx=0, dy=1, ksize=3)

thr = 100  # mais ou menos contornos

imgMag = np.sqrt(cv2.add(np.float32(imgDx) ** 2, np.float32(imgDy) ** 2))
imgPhase = np.arctan2(imgDx, imgDy)
imgPhaseNorm = np.uint8((imgPhase / np.pi / 2 + 1) * 255)

imgE = np.uint8(imgMag / (np.max(imgMag) + np.finfo(np.float32).resolution) * 255)

# contours = np.int16((cv2.convertScaleAbs(imgE) > thr) * 255)
# imgDxCont = cv2.multiply(imgDx, contours, scale=1.0/255)
# imgDyCont = cv2.multiply(imgDy, contours, scale=1.0/255)
# plt.quiver(imgDxCont, imgDyCont)
# plt.show()

imggauss = cv2.GaussianBlur(imgg, (7, 7), 1)
imgECanny = cv2.Canny(imggauss, thr / 2, thr)

cv2.imshow('dx Img', cv2.convertScaleAbs(imgDx))
cv2.imshow('dy Img', cv2.convertScaleAbs(imgDy))

cv2.imshow('Gradient Mag', cv2.convertScaleAbs(imgE))
cv2.imshow('Gradient Mag Thresh', np.uint8(cv2.convertScaleAbs(imgE) > thr) * 255)
cv2.imshow('Gradient Phase', imgPhaseNorm)

cv2.imshow('Canny', imgECanny)

cv2.waitKey(0)
cv2.destroyAllWindows()
