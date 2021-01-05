import cv2
import numpy as np
from libs import bwLabel, psColor

fileDir = 'imageDatabase/'
fileName = 'ImTest1.jpg'

img = cv2.imread(fileDir + fileName)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Image', imgg)

thres, bw = cv2.threshold(imgg, 127, 255, cv2.THRESH_OTSU)
print(thres)

cv2.imshow('BW Image', bw)

strElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))
bw1 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, strElem)

cv2.imshow('BW Morph Image', bw1)

regionNum, lb = cv2.connectedComponents(bw1)
cv2.imshow('Label bw', np.uint8(lb * np.round(255.0 / regionNum)))

colorM = psColor.CreateColorMap(regionNum, 1)
pseudoC = psColor.Gray2PseudoColor(lb, colorM)

cv2.imshow('Label Image', pseudoC)

cv2.waitKey(0)
cv2.destroyAllWindows()
