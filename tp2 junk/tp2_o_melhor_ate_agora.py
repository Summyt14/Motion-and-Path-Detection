from scipy.spatial import distance as dist

from cv2 import cv2
import numpy as np

frame = None
backup_frame = None
aof = None
aof_points = []
classes=['Person', 'Car', 'Other']
classColor = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
trajetoria = {}
lastPos={}
nextObjectID = 0
disappeared = {}
maxDisappeared = 150
ind = -4
id_n = 0


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


def define_type(x,y,w, h, id, outro):
    global trajetoria, aof
    #  ====== Os ifs podem precisar de mais condicoes ======
    if h > w:
        cv2.putText(frame, classes[0], (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[0], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[0], 2)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[0],  2, cv2.LINE_AA)

    elif w > h:
        cv2.putText(frame, classes[1], (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[1], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[1], 2)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[1], 2, cv2.LINE_AA)

    elif outro:
        cv2.putText(frame, classes[2], (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[2], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[2], 2)
        cv2.drawMarker(aof, ((x + w) // 2, (y + h) // 2), classColor[2], cv2.MARKER_STAR, 1, 1)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[2], 2, cv2.LINE_AA)

    if trajetoria.get(id) is not None:
        trajetoria[id].append((x + (w // 2), h + y))
    else:
        trajetoria[id] = [(int(x + (w // 2)), int(h + y))]


def classify_contour(contours):
    global ind, id_n
    inputCentroids = np.zeros((len(contours), 2), dtype='int')
    idx = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        idx += 1

        if area > 400 and ind > -1:
            centroid_X = x+(w//2)
            centroid_Y = y+(h//2)
            print(idx)
            inputCentroids[idx] = (centroid_X, centroid_Y)
            print(inputCentroids)

            if len(lastPos) == 0:
                for j in range(0, len(inputCentroids)):
                    lastPos[id_n] = (centroid_X, centroid_Y)
                    disappeared[id_n] = 0
                    id_n+=1
                    define_type(x, y, w, h, 0, False)
            else:
                id_s = list(disappeared.keys())
                encontrado = False
                for id in id_s:
                    elem = lastPos.get(id)
                    for i in range(-30, 30):
                        for j in range(-30, 30):
                            if elem == (centroid_X + i, centroid_Y + j):
                                lastPos[id] = (centroid_X, centroid_Y)
                                disappeared[id] = 0
                                encontrado = True
                                define_type(x, y, w, h, id, False)
                                break
                        if encontrado:
                            break
                    if encontrado:
                        break
                if not encontrado:
                    id_n = list(lastPos.keys())[-1] + 1
                    lastPos[id_n] = (centroid_X, centroid_Y)
                    disappeared[id_n] = 0
                    define_type(x, y, w, h, id_n, False)
        ind+=1

def save_trajectories(spot):
    global nextObjectID, lastPos, trajetoria
    lastPos[nextObjectID] = spot
    disappeared[nextObjectID] = 0
    nextObjectID+=1


def remove_trajectories(id):
    global lastPos, disappeared, trajetoria
    if disappeared.get(id) is not None:
        disappeared.pop(id)
    if trajetoria.get(id) is not None:
        trajetoria.pop(id)


def show_trajectories():
    global trajetoria, aof
    if len(trajetoria) == 0:
        return None

    for j in trajetoria.keys():
        for i in range(1, len(trajetoria.get(j))):
            if trajetoria.get(j)[i] is None or trajetoria.get(j)[i-1] is None:
                continue
            cv2.line(aof, trajetoria.get(j)[i-1], trajetoria.get(j)[i], (j*30,250,j*2+100), 2)

    for m in list(disappeared.keys()):
        disappeared[m] = disappeared[m] + 1
        if disappeared[m] > maxDisappeared:
            remove_trajectories(m)
    # mostrar as trajetorias (uma linha a unir os pontos) de cada objeto
    # uma cor com id para cada um
    # mostrar trajetoria como se fosse video ou ja tudo desenhado?


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

        classify_contour(contours)

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
