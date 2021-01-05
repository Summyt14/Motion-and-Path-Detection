import cv2
import numpy as np
import matplotlib.pyplot as plt

fileDir = 'imageDatabase/'
fileName = 'JonquilFlowers.jpg'
# fileName = 'cameraman.jpg'

img = cv2.imread(fileDir + fileName)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', imgg)

# sobel
kernelx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

kernely = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

img_prewittx = cv2.filter2D(imgg, cv2.CV_16S, kernelx)
img_prewitty = cv2.filter2D(imgg, cv2.CV_16S, kernely)

cv2.imshow('dx Img', cv2.convertScaleAbs(img_prewittx))
cv2.imshow('dy Img', cv2.convertScaleAbs(img_prewitty))

cv2.waitKey(0)
cv2.destroyAllWindows()
