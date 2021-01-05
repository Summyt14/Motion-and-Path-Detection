import cv2

fileDir = 'imageDatabase/'

alpha = 0.5

foreground = cv2.imread(fileDir + "falcon.jpg")
# background = cv2.imread(fileDir + "florest.jpg")
background = cv2.imread(fileDir + "lily-lotus-flowers.jpg")

# dst = cv2.addWeighted(foreground, alpha, background, 1-alpha, 0.0)

resized_img = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
dst = cv2.addWeighted(foreground, alpha, resized_img, 1 - alpha, 0.0)

cv2.imshow('dst', dst)

# cv2.imwrite(fileDir + "weighted.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
