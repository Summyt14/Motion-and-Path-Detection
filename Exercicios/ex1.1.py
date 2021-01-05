import cv2

fileDir = 'imageDatabase/'
fileName = 'lily-lotus-flowers.jpg'

img = cv2.imread(fileDir + fileName)

# cv2.namedWindow('Image', 1)
cv2.imshow('Color Image', img)
cv2.imshow('Blue Component', img[:, :, 0])
cv2.imshow('Green Component', img[:, :, 1])
cv2.imshow('Red Component', img[:, :, 2])

cv2.waitKey(0)
cv2.destroyAllWindows()
