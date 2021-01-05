import cv2

fileDir = 'imageDatabase/'

foreground = cv2.imread(fileDir + "falcon.jpg").astype(float) / 255
background = cv2.imread(fileDir + "florest.jpg").astype(float) / 255
mask = cv2.imread(fileDir + "mask.png").astype(float) / 255

foreground = cv2.multiply(foreground, mask)
background = cv2.multiply(background, 1 - mask)
# background = cv2.multiply(background, 255 - mask, scale=1.0 / 255)
result = cv2.add(foreground, background)

cv2.imshow("Image", result)

cv2.waitKey(0)
cv2.destroyWindow("Image")
