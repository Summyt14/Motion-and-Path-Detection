
from cv2 import cv2
import numpy as np

frame = None
backup_frame = None
aof = None
aof_points = []
classes = ['Person', 'Car', 'Other']
classColor = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
trajetoria = {}
lastPos = {}
nextObjectID = 0
disappeared = {}
maxDisappeared = 115
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


def save_trajectories(x, y, w, h, id):
    if trajetoria.get(id) is not None:
        trajetoria[id].append((x + (w // 2), h + y))
    else:
        trajetoria[id] = [(int(x + (w // 2)), int(h + y))]

def mark_id(x, y, w, h, id):
    if h / w > 1.2:
        cv2.putText(frame, classes[0], (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[0], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[0], 2)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[0], 2, cv2.LINE_AA)
    elif w > h:
        cv2.putText(frame, classes[1], (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[1], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[1], 2)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[1], 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, classes[2], (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[2], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), classColor[2], 2)
        cv2.drawMarker(aof, ((x + w) // 2, (y + h) // 2), classColor[2], cv2.MARKER_STAR, 1, 1)
        cv2.putText(aof, str(id), (x + (w // 2), h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, classColor[2], 2, cv2.LINE_AA)


def define_type(x, y, w, h, id, isToMark_text):
    global trajetoria, aof
    #  ====== Os ifs podem precisar de mais condicoes ======
    if h / w > 1.2:
        type = 1
    elif w > h:
        type = 2
    else:
        type = 3
    if isToMark_text:
        mark_id(x, y, w, h, id)
    return type

def classify_contour(contours):
    global ind, id_n
    inputCentroids = np.zeros((len(contours), 2), dtype='int')
    idx = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        idx += 1

        if area > 400 and ind > -1:
            centroid_X = x + (w // 2)
            centroid_Y = y + (h // 2)
            inputCentroids[idx] = (centroid_X, centroid_Y)

            if len(lastPos) == 0:
                for j in range(0, len(inputCentroids)):
                    type = define_type(x, y, w, h, id_n, True)
                    lastPos[id_n] = (centroid_X, centroid_Y, type)
                    disappeared[id_n] = 0
                    save_trajectories(x, y, w, h, id_n)

            else:
                id_s = list(disappeared.keys())
                encontrado = False
                for id in id_s:
                    elem = lastPos.get(id)
                    #type = define_type(x, y, w, h, id, False)
                    for i in range(-50, 50):
                        for j in range(-20, 20):
                            if elem[0] == centroid_X + i and elem[1] == centroid_Y + j:
                                disappeared[id] = 0
                                if centroid_X + i < 17 or centroid_X + i > 750:
                                    remove_trajectories(id)
                                    return
                                encontrado = True
                                define_type(x, y, w, h, id, True)
                                lastPos[id] = (centroid_X, centroid_Y, elem[2])
                                save_trajectories(x, y, w, h, id)
                                break
                        if encontrado:
                            break
                    if encontrado:
                        break
                if not encontrado:
                    id_n = list(lastPos.keys())[-1] + 1
                    if centroid_X < 17 or centroid_X > 750:
                        remove_trajectories(id_n)
                        return
                    type = define_type(x, y, w, h, id_n, True)
                    lastPos[id_n] = (centroid_X, centroid_Y, type)
                    disappeared[id_n] = 0
                    save_trajectories(x, y, w, h, id_n)

        ind += 1


#### colors = [aof[centroid] for centroid in centroids]

def remove_trajectories(id):
    global disappeared, trajetoria
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
            if trajetoria.get(j)[i] is None or trajetoria.get(j)[i - 1] is None:
                continue
            cv2.line(aof, trajetoria.get(j)[i - 1], trajetoria.get(j)[i], (j * 30, 250, j * 2 + 100), 2)

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
    object_detector = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    ret, frame = cap.read()
    backup_frame = frame.copy()

    choose_area_of_effect()
    print('AOF:', aof_points)

    ret, frame = cap.read()
    while cap.isOpened():
        aof = frame[aof_points[2]:aof_points[3], aof_points[0]:aof_points[1]]

        mask = object_detector.apply(aof)
        median = cv2.medianBlur(mask, 3)
        thresh = cv2.Canny(median, 210, 255)
        #_, thresh = cv2.threshold(median, 200, 255, cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4), (-1, -1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (-1, -1))
        #dilation = cv2.dilate(thresh, kernel1, iterations=1)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
        #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
