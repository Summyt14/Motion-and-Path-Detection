from cv2 import cv2
import numpy as np

frame = None
backup_frame = None
aof = None
aof_points = []
classes=['Person', 'Car', 'Other']
classColor = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
trajetoria = {0:[], 1:[], 2:[]}
lastPos=[]

def draw_rect(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        aof_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(backup_frame, (x, y), 5, (0, 0, 255), -1)


def choose_area_of_effect():
    global frame, backup_frame, aof_points, aof

    while len(aof_points) < 2:

        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', draw_rect)

        if len(aof_points) == 0:
            cv2.putText(frame, 'Defina o canto superior esquerdo da frame de detecao', (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            frame = backup_frame
            cv2.putText(frame, 'Defina o canto inferior direito da frame de detecao', (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(40) == 27:  # Escape Btn
            break

    # Start Width, End Width, Start Height, End Height
    aof_points = [aof_points[0][0]] + [aof_points[1][0]] + [aof_points[0][1]] + [aof_points[1][1]]
    cv2.destroyAllWindows()


def classify_contour(frame, aof, contour):
    area = cv2.contourArea(contour)

    if area > 400:
        (x, y, w, h) = cv2.boundingRect(contour)
        # save_trajectories(x,y)
        #  ====== Os ifs podem precisar de mais condicoes ======
        if h > w:
            cv2.putText(frame, classes[0], (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[0], 2, cv2.LINE_AA)
            cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[0], 2)
            cv2.circle(aof, (x+(w//2), h+y), 1, (0,0,0), 10)
            trajetoria[0].append((x+(w//2), h+y))

        elif w > h:
            cv2.putText(frame, classes[1], (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[1], 2, cv2.LINE_AA)
            cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[1], 2)
            cv2.circle(aof, (x+(w//2), h+y), 1, (0,0,0), 10)
            trajetoria[1].append((x+(w//2), h+y))

        else:
            cv2.putText(frame, classes[2], (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[2], 2, cv2.LINE_AA)
            cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[2], 2)
            cv2.drawMarker(aof, ((x+w)//2, (y+h)//2), classColor[2],  cv2.MARKER_STAR, 1, 1)
            cv2.circle(aof, (x+(w//2), h+y), 1, (0,0,0), 10)
            trajetoria[2].append((x+(w//2), h+y))


def save_trajectories(x,y):
    # this is not working
    repetido=False
    id=-1
    for elem in lastPos:
        for i in range(-10, 10):
            for j in range(-10, 10):
                if elem == (x + i,y + j):
                    found = (x + i, y + j)
                    id = lastPos.index(elem)
                    lastPos[id] = found
                    print(id, x,y)
                    repetido=True
                    break
    if not repetido:
        lastPos.append((x,y))
        id = lastPos.index((x,y))
    # confirmar se é o msm objeto que nao foi detetado numa frame
    # guardar numa variavel global
    return None


def show_trajectories():
    global trajetoria, aof
    for j in range(len(trajetoria.items())):
        for i in range(1, len(trajetoria.get(j))):
            if trajetoria.get(j)[i - 1] is None or trajetoria.get(j)[i] is None:
                continue
            cv2.line(aof, trajetoria.get(j)[i-1], trajetoria.get(j)[i], (j*100,0,j*2+50), 2)
    # mostrar as trajetorias (uma linha a unir os pontos) de cada objeto
    # uma cor com id para cada um
    # mostrar trajetoria como se fosse video ou ja tudo desenhado?
    return None


def motion_tracker(video):
    global frame, backup_frame, aof
    cap = cv2.VideoCapture(video)
    object_detector = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    ret, frame = cap.read()
    backup_frame = frame.copy()

    choose_area_of_effect()
    print('AOF:', aof_points)

    ret, frame = cap.read()
    while cap.isOpened():
        aof = frame[aof_points[2]:aof_points[3], aof_points[0]:aof_points[1]]

        mask = object_detector.apply(aof)
        median = cv2.medianBlur(mask, 255 // 26)
        _, thresh = cv2.threshold(median, 200, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4), (-1, -1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (-1, -1))
        dilation = cv2.dilate(thresh, kernel1, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            classify_contour(frame, aof, contour)

        show_trajectories()
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', closing)
        cv2.imshow('AOF', aof)
        ret, frame = cap.read()

        if cv2.waitKey(40) == 27 or frame is None:  # Escape Btn
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    motion_tracker('camera1.mp4')
