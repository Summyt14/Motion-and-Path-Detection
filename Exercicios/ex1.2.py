import cv2

# ID da camera
video_src = 0
cam = cv2.VideoCapture(video_src)

# cv2.namedWindow('Video', 0)
# strElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (-1, -1))

while True:
    flag, img = cam.read()
    # print(flag)
    if flag:
        # imgg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # thresh, imgBW = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY)
        # imgE = cv2.medianBlur(img, 17)
        # imgE = cv2.blur(img, (17, 17))
        imgE = cv2.Canny(img, 100, 200)
        # imgMorph = cv2.dilate(imgg, strElem)
        # imgMorph = cv2.dilate(imgBW, strElem)
        cv2.imshow('Video', imgE)

    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        cam.release()
        break
