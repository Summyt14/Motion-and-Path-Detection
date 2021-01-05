import cv2

fileDir = 'imageDatabase/'
fileName = 'JonquilFlowers.jpg'

img = cv2.imread(fileDir + fileName)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Image', img)

thres, bw = cv2.threshold(imgg, 140, 255, cv2.THRESH_BINARY)
# print(thres)

cv2.imshow('BW Image', bw)

strElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))

# bw1 = cv2.dilate(bw, strElem)
# bw2 = cv2.erode(bw, strElem)
bw1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, strElem)
bw2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, strElem)

cv2.imshow('BW Morph Image 1', bw1)
cv2.imshow('BW Morph Image 2', bw2)

# outFileName = 'coins1BW.png'
# cv2.imwrite(outFileName, bw1)

cv2.waitKey(0)
cv2.destroyAllWindows()
