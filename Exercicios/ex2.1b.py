import cv2
import numpy as np

# Este codigo cria a mascara sabendo apenas a cor do fundo (neste caso verde)

fileDir = 'imageDatabase/'
objFileName = 'falcon.jpg'
bgFileName = 'florest.jpg'

chromaKey_color = np.array([101, 236, 192])

objImage = cv2.imread(fileDir + objFileName)
bgImage = cv2.imread(fileDir + bgFileName)

maskImageInv = cv2.inRange(objImage, chromaKey_color - 20, chromaKey_color + 20)
maskImageM = 255 - maskImageInv
cv2.imshow('Mask Original', maskImageM)

kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))
maskImageMM = cv2.morphologyEx(maskImageM, cv2.MORPH_OPEN, kernel1)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (-1, -1))
maskImageMM2 = cv2.morphologyEx(maskImageMM, cv2.MORPH_OPEN, kernel2)
cv2.imshow('Mask', maskImageMM2)

maskImage = cv2.cvtColor(maskImageMM2, cv2.COLOR_GRAY2BGR)

img1 = cv2.multiply(objImage, maskImage, scale=1.0 / 255)
img2 = cv2.multiply(bgImage, 255 - maskImage, scale=1.0 / 255)

outImage = cv2.add(img1, img2)

cv2.imshow('Image Out', outImage)
cv2.imshow('Image Part 1', img1)
cv2.imshow('Image Part 2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
