import cv2

fileDir = 'imageDatabase/'
fileName = 'fingerprint.jpg'

img = cv2.imread(fileDir + fileName)

print('Original Dimensions : ', img.shape)

Ifx = 0.1
Ify = 0.1
# resize image
resized = cv2.resize(img, None,fx=Ifx, fy=Ify, interpolation=cv2.INTER_NEAREST)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Original image", img)
cv2.imshow("Resized image", resized)
cv2.imwrite(fileDir + 'fingerprint2.jpg', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
