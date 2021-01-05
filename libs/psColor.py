# filename: psColor Module
import math
import cv2
import numpy as np


def Gray2PseudoColor(grayImage, colorMap):
    imageDim = grayImage.shape
    colorImage = np.zeros((imageDim[0], imageDim[1], 3), dtype=np.uint8)
    colormapDim = colorMap.shape
    for i in range(colormapDim[0]):
        inds = np.nonzero(grayImage == i)
        colorImage[inds] = (colorMap[i, 0], colorMap[i, 1], colorMap[i, 2])
    return colorImage


def CreateColorMap(colorNum, firstColorBlack=0):
    colorMap = np.zeros((colorNum, 3), dtype=np.uint8)
    for i in range(colorNum):
        colorMap[i, 0] = int(round(255 * abs(math.sin(1.0 * i * math.pi / 2 / colorNum))))
        colorMap[i, 1] = int(round(255 * abs(math.sin(1.0 * i * math.pi / 2 / colorNum + math.pi / 2))))
        colorMap[i, 2] = int(round(255 * abs(math.sin(1.0 * i * math.pi / colorNum))))
    if firstColorBlack:
        colorMap[0, 0] = 0
        colorMap[0, 1] = 0
        colorMap[0, 2] = 0
    return colorMap
