import cv2

fileDir = 'imageDatabase/'

img = cv2.imread(fileDir + "falcon.jpg")

rotMatrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 180, 1)

# para 90º
# rotMatrix = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 90, 1)
# rotMatrix[0][2] = 0
# rotMatrix[1][2] = 399
# imgFinal = cv2.warpAffine(img, rotMatrix, (img.shape[0], img.shape[1]))

imgFinal = cv2.warpAffine(img, rotMatrix, (img.shape[1], img.shape[0]))

# cv2.imwrite("rotFalcon.jpg", imgFinal)
cv2.imshow('Rotated', imgFinal)
cv2.waitKey(0)
cv2.destroyAllWindows()
