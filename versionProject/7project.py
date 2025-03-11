import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para calcular la distancia entre dos puntos
def calcular_distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

# Función para detectar la seña de 'universidad'
def detect_university_sign(landmarks):
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]

    if (index_finger.y < landmarks[7].y and pinky_finger.y < landmarks[19].y):
        thumb_finger = landmarks[4]
        middle_finger = landmarks[12]
        ring_finger = landmarks[16]

        if (thumb_finger.y > landmarks[3].y and middle_finger.y > landmarks[11].y and ring_finger.y > landmarks[15].y):
            return True
    return False

# Función para verificar si la palma está abierta
def dedos_extremos(landmarks):
    return all(
        landmarks.landmark[finger_tip].y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
        for finger_tip in [
            mp_holistic.HandLandmark.THUMB_TIP,
            mp_holistic.HandLandmark.INDEX_FINGER_TIP,
            mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
            mp_holistic.HandLandmark.RING_FINGER_TIP,
            mp_holistic.HandLandmark.PINKY_TIP,
        ]
    )

# Función para detectar "Amigo"
def detectar_amigo(landmarks_left, landmarks_right):
    if landmarks_left and landmarks_right:
        wrist_left = landmarks_left.landmark[mp_holistic.HandLandmark.WRIST]
        wrist_right = landmarks_right.landmark[mp_holistic.HandLandmark.WRIST]

        dist_wrist = abs(wrist_left.x - wrist_right.x) + abs(wrist_left.y - wrist_right.y)

        if dist_wrist < 0.2 and dedos_extremos(landmarks_left) and dedos_extremos(landmarks_right):
            return "Amigo"
    return None

# Función para verificar si la palma está hacia abajo
def hacer(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist_y = landmarks[0].y
    knuckles_y = sum([landmarks[i].y for i in [5, 9, 13, 17]]) / 4
    return wrist_y < knuckles_y

# Función para verificar si los nudillos están por encima de la muñeca (para Casa y Correr)
def nudillos_sobre_muneca(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist_y = landmarks[mp_holistic.HandLandmark.WRIST].y
    knuckles_y = sum([
        landmarks[i].y for i in [
            mp_holistic.HandLandmark.INDEX_FINGER_MCP,
            mp_holistic.HandLandmark.MIDDLE_FINGER_MCP,
            mp_holistic.HandLandmark.RING_FINGER_MCP,
            mp_holistic.HandLandmark.PINKY_MCP
        ]
    ]) / 4
    return knuckles_y < wrist_y

# Función para detectar "Hola"

def dedos_separados(hand_landmarks):
    landmarks = hand_landmarks.landmark
    dist_index_middle = calcular_distancia(landmarks[8], landmarks[12])
    dist_middle_ring = calcular_distancia(landmarks[12], landmarks[16])
    dist_ring_pinky = calcular_distancia(landmarks[16], landmarks[20])
    
    # Verificar que las distancias sean significativas para considerar los dedos separados
    return dist_index_middle > 0.05 and dist_middle_ring > 0.05 and dist_ring_pinky > 0.05

# Función para detectar "Hola"
def detectar_hola(results):
    def is_palm_open(hand_landmarks):
        if not hand_landmarks:
            return False
        
        landmarks = hand_landmarks.landmark
        
        # Verificar si los dedos están extendidos (yema por debajo de la falange media)
        dedos_extendidos = all(
            landmarks[finger_tip].y < landmarks[finger_dip].y
            for finger_tip, finger_dip in [
                (8, 6),   # Pulgar (yema vs falange media)
                (12, 10), # Índice (yema vs falange media)
                (16, 14), # Medio (yema vs falange media)
                (20, 18)  # Anular (yema vs falange media)
            ]
        )
        
        # Verificar que la palma está abierta y que los dedos están separados
        # Se asume que si los dedos están extendidos y separados, la palma está abierta
        palma_abierta = dedos_extendidos and dedos_separados(hand_landmarks)
        
        return palma_abierta

    # Detectar si la palma abierta y dedos parados están en la mano izquierda o derecha
    if is_palm_open(results.left_hand_landmarks) or is_palm_open(results.right_hand_landmarks):
        return "Hola"
    
    return None


# Función para detectar "Correr"
def detectar_correr(results):
    if results.left_hand_landmarks and results.right_hand_landmarks:
        if nudillos_sobre_muneca(results.left_hand_landmarks) and nudillos_sobre_muneca(results.right_hand_landmarks):
            return "Correr"
    return None

# Función para detectar "Casa"
def detectar_gesto_casa(results):
    if results.left_hand_landmarks and results.right_hand_landmarks:
        left_hand = results.left_hand_landmarks.landmark
        right_hand = results.right_hand_landmarks.landmark

        left_index = (left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                      left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
        right_index = (right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                       right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)

        distance = math.sqrt((left_index[0] - right_index[0])**2 + (left_index[1] - right_index[1])**2)

        if distance < 0.1 and nudillos_sobre_muneca(results.left_hand_landmarks) and nudillos_sobre_muneca(results.right_hand_landmarks):
            return "Casa"
    return None

# Función para detectar "Comer"
def detect_comer(landmarks, boca):
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]
    return calcular_distancia(index_finger, boca) < 0.1 or calcular_distancia(pinky_finger, boca) < 0.1

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        mensaje = "No detectado"

        if results.face_landmarks:
            boca = results.face_landmarks.landmark[13]  # Centro de la boca

            if results.left_hand_landmarks and detect_comer(results.left_hand_landmarks.landmark, boca):
                mensaje = "Comer"

            if results.right_hand_landmarks and detect_comer(results.right_hand_landmarks.landmark, boca):
                mensaje = "Comer"

        mensaje = detectar_hola(results) or mensaje
        mensaje = detectar_correr(results) or mensaje
        mensaje = detectar_gesto_casa(results) or mensaje
        mensaje = detectar_amigo(results.left_hand_landmarks, results.right_hand_landmarks) or mensaje

        if results.left_hand_landmarks and results.right_hand_landmarks:
            if hacer(results.left_hand_landmarks) and hacer(results.right_hand_landmarks):
                mensaje = "Hacer"

        if results.left_hand_landmarks and detect_university_sign(results.left_hand_landmarks.landmark):
            mensaje = "Universidad"

        if results.right_hand_landmarks and detect_university_sign(results.right_hand_landmarks.landmark):
            mensaje = "Universidad"

        cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Gestos Detectados', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
