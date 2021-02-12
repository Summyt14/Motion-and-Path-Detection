from cv2 import cv2
import numpy as np
import numpy.random as rd

frame = None
backup_frame = None
aof = None
aof_points = []
classes = ['Person', 'Car', 'Other']
class_color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
trajectories = {}
trajectories_color = {}
last_pos = {}
disappeared_timer = {}
max_time_disappeared = 115
max_trajectory_size = 150
starting_frames = -4
new_obj_id = 0


def draw_rect(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Guarda as coordenadas do clique
        aof_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.circle(backup_frame, (x, y), 5, (0, 0, 255), -1)


def choose_area_of_effect():
    global frame, backup_frame, aof_points, aof

    # Loop ate ter 2 pontos escolhidos
    while len(aof_points) < 2:
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', draw_rect)

        # Primeiro ponto
        if len(aof_points) == 0:
            cv2.putText(frame, 'Defina o canto superior esquerdo da frame de detecao', (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # Segundo ponto
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


def define_type(x, y, w, h, obj_id):
    global aof

    # Pessoa
    if h / w > 1.2:
        obj_type = 1
        cv2.putText(frame, classes[0], (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[0], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), class_color[0], 2)
        cv2.putText(aof, str(obj_id), (x + (w // 3), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[0], 2,
                    cv2.LINE_AA)
    # Carro
    elif w > h:
        obj_type = 2
        cv2.putText(frame, classes[1], (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[1], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), class_color[1], 2)
        cv2.putText(aof, str(obj_id), (x + (w // 3), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[1], 2,
                    cv2.LINE_AA)
    # Outro
    else:
        obj_type = 3
        cv2.putText(frame, classes[2], (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[2], 2, cv2.LINE_AA)
        cv2.rectangle(aof, (x, y), (x + w, y + h), class_color[2], 2)
        cv2.drawMarker(aof, ((x + w) // 2, (y + h) // 2), class_color[2], cv2.MARKER_STAR, 1, 1)
        cv2.putText(aof, str(obj_id), (x + (w // 3), y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color[2], 2,
                    cv2.LINE_AA)

    return obj_type


def save_trajectories(x, y, w, h, obj_id):
    # Adiciona pontos a trajetoria de um objeto
    if trajectories.get(obj_id) is not None:
        trajectories[obj_id].append((x + (w // 2), h + y))

    # Cria uma nova trajetoria se nao existir uma para esse objeto
    else:
        trajectories[obj_id] = [(int(x + (w // 2)), int(h + y))]
        trajectories_color[obj_id] = (rd.randint(0, 255), rd.randint(0, 255), rd.randint(0, 255))


def remove_trajectories(obj_id):
    # Remove o contador de tempo desaparecido do objeto
    if disappeared_timer.get(obj_id) is not None:
        disappeared_timer.pop(obj_id)
    # Remove a trajetoria do objeto
    if trajectories.get(obj_id) is not None:
        trajectories.pop(obj_id)
        trajectories_color.pop(obj_id)


def show_trajectories():
    global aof
    if len(trajectories) == 0:
        return None

    # Para cada trajetoria:
    for i in trajectories.keys():
        # Para cada ponto de uma trajetoria:
        for j in range(1, len(trajectories.get(i))):
            # Nao se faz nada se nao houver pontos ou um ponto anterior
            if trajectories.get(i)[j] is None or trajectories.get(i)[j - 1] is None:
                continue
            # Desenha uma linha a unir o ponto atual ao ponto anterior
            cv2.line(aof, trajectories.get(i)[j - 1], trajectories.get(i)[j],
                     (trajectories_color.get(i)[0], trajectories_color.get(i)[1], trajectories_color.get(i)[2]), 2)

    for i in list(disappeared_timer.keys()):
        disappeared_timer[i] = disappeared_timer[i] + 1
        if disappeared_timer[i] > max_time_disappeared:
            remove_trajectories(i)


def classify_contour(contours):
    global starting_frames, new_obj_id
    centroids = np.zeros((len(contours), 2), dtype='int')
    idx = -1

    # Para cada contorno:
    for contour in contours:
        # Calcula-se a area do contorno
        area = cv2.contourArea(contour)
        # Guarda-se as dimensoes do rectangulo que encaixa no contorno
        (x, y, w, h) = cv2.boundingRect(contour)
        idx += 1

        # Se a area for maior que 400 e se se estiver nas frames iniciais para o calculo:
        if area > 400 and starting_frames > -1:
            # Calcula-se os centroides do objeto
            centroid_X = x + (w // 2)
            centroid_Y = y + (h // 2)
            # Adiciona-se o centroide a lista de centroides de regioes ativas
            centroids[idx] = (centroid_X, centroid_Y)
            # Se não houver nenhuma região ativa na cena
            if len(last_pos) == 0:
                for _ in centroids:
                    # Classificacao
                    obj_type = define_type(x, y, w, h, new_obj_id)
                    # Adiciona-se a lista de posicoes das regioes ativas
                    last_pos[new_obj_id] = (centroid_X, centroid_Y, obj_type)
                    # Coloca-se o tempo de desaparecimento respetivo a zero
                    disappeared_timer[new_obj_id] = 0
                    # Adiciona-se a posicao a lista de trajetorias
                    save_trajectories(x, y, w, h, new_obj_id)
            # Caso a cena tenha regioes ativas
            else:
                # Flag de encontrar correspondencia com regiao ativa
                found = False
                # Procurar em cada elemento das regioes ativas
                for obj_id in list(disappeared_timer.keys()):
                    elem = last_pos.get(obj_id)
                    # Se em x
                    for i in range(-50, 50):
                        # E em y
                        for j in range(-20, 20):
                            # Se um dos elementos esta a uma distancia proxima da atual, e o mesmo elemento
                            if elem[0] == centroid_X + i and elem[1] == centroid_Y + j:
                                # Entao reinicia-se o respetivo timer
                                disappeared_timer[obj_id] = 0
                                # Se a posicao for nas margens do campo de visao da camera
                                if centroid_X + i < 17 or centroid_X + i > 750:
                                    # Retira-se o elemento, pois saiu do campo de visao
                                    remove_trajectories(obj_id)
                                    return
                                # Mudanca do valor booleano da flag pois houve correspondencia
                                found = True
                                # Classificacao do elemento
                                define_type(x, y, w, h, obj_id)
                                # Guardar a sua posicao atual
                                last_pos[obj_id] = (centroid_X, centroid_Y, elem[2])
                                save_trajectories(x, y, w, h, obj_id)
                                break
                        if found:
                            break
                    if found:
                        break
                # Se nao tiver havido correspondencia, significa que e um novo objeto
                if not found:
                    # Adiciona-se as estruturas de dados, com um novo identificador
                    new_obj_id = list(last_pos.keys())[-1] + 1
                    if centroid_X < 17 or centroid_X > 750:
                        remove_trajectories(new_obj_id)
                        return
                    obj_type = define_type(x, y, w, h, new_obj_id)
                    last_pos[new_obj_id] = (centroid_X, centroid_Y, obj_type)
                    disappeared_timer[new_obj_id] = 0
                    save_trajectories(x, y, w, h, new_obj_id)
        starting_frames += 1


def motion_tracker(video):
    global frame, backup_frame, aof
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    backup_frame = frame.copy()

    choose_area_of_effect()

    object_detector = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    ret, frame = cap.read()
    while cap.isOpened():
        aof = frame[aof_points[2]:aof_points[3], aof_points[0]:aof_points[1]]

        mask = object_detector.apply(aof)
        median = cv2.medianBlur(mask, 3)
        thresh = cv2.Canny(median, 210, 255)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (-1, -1))
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
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
