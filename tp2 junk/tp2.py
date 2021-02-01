import glob
import cv2
import numpy as np


# Function to extract frames
def save_frames(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1

    while success:
        success, image = vidObj.read()
        cv2.imwrite("frames/frame%d.jpg" % count, image)
        count += 1


def motion_tracker(imgs):
    img1 = cv2.imread(imgs[0])
    img2 = cv2.imread(imgs[1])
    idx = 2

    while idx < len(imgs):
        print(idx)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # median = cv2.medianBlur(img1, 255//35)

        diff = cv2.absdiff(img1_gray, img2_gray)

        # image = cv2.resize(diff, (1280, 720))
        # out.write(image)
        cv2.imshow("feed", diff)
        img1 = img2
        img2 = cv2.imread(imgs[idx])
        idx += 1

        if cv2.waitKey(40) == 27:
            break


if __name__ == '__main__':
    img_path = glob.glob("frames/*.jpg")
    img_path.sort()

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

    out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

    motion_tracker(img_path)
