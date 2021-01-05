import cv2

video_src = 0
cam = cv2.VideoCapture(video_src)

while True:
    flag, camImg = cam.read()
    if flag:
        camImgBlur = cv2.blur(camImg, (3, 33))
        camImgBlur2 = cv2.medianBlur(camImg, 255)
        camImgBlur3 = cv2.GaussianBlur(camImg, (3, 33), 0)
        cv2.imshow("Original Image", camImg)
        cv2.imshow("Blured Image", camImgBlur)
        cv2.imshow("MedianBlured Image", camImgBlur2)
        cv2.imshow("GaussianBlured Image", camImgBlur3)

    # Press Esc to exit
    key = cv2.waitKey(33)
    if key == 27:
        cv2.destroyAllWindows()
        cam.release()
        break
