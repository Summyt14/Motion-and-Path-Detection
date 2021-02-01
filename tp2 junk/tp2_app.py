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


def motion_tracker(path):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        img1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(img1_gray, img2_gray)

        median = cv2.medianBlur(diff, 255 // 50)
        _, thresh = cv2.threshold(median, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # if len(contours) > 0:
        #     rect = cv2.minAreaRect(contours)
        #     box = cv2.boxPoints(rect)
        #     box = np.int0(box)
        #     cv2.drawContours(frame1, [box], 0, (0, 0, 255), 2)

        image = cv2.resize(diff, (1280, 720))
        out.write(image)
        cv2.imshow("feed", diff)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == '__main__':
    # img_path = glob.glob("frames/*.jpg")
    # img_path.sort()

    motion_tracker("camera1.mp4")
