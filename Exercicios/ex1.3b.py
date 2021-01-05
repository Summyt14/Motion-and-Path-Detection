import cv2

fileDir = 'imageDatabase/'
fileName2 = 'fingerprint2.jpg'

img2 = cv2.imread(fileDir + fileName2)

Ifx = 10
Ify = 10
# resize image
resized1 = cv2.resize(img2, (500, 500), fx=Ifx, fy=Ify, interpolation=cv2.INTER_CUBIC)
resized2 = cv2.resize(img2, (500, 300), fx=Ifx, fy=Ify, interpolation=cv2.INTER_NEAREST)
resized3 = cv2.resize(img2, (300, 500), fx=Ifx, fy=Ify, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Original image", img2)
cv2.imshow("Cubic image", resized1)
cv2.imshow("Nearest image", resized2)
cv2.imshow("Linear image", resized3)
cv2.waitKey(0)
cv2.destroyAllWindows()
