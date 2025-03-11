import cv2
import mediapipe as mp

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para detectar la seña de 'universidad' basada en los dedos índice y meñique alzados
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

# Función para verificar si los dedos están extendidos (mano abierta)
def dedos_extremos(landmarks):
    thumb_tip = landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

    if (thumb_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        index_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        middle_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        ring_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        pinky_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y):
        return True
    return False

# Función para detectar el gesto de "amigo"
def detectar_amigo(landmarks_left, landmarks_right):
    if landmarks_left and landmarks_right:
        wrist_left = landmarks_left.landmark[mp_holistic.HandLandmark.WRIST]
        wrist_right = landmarks_right.landmark[mp_holistic.HandLandmark.WRIST]

        dist_wrist = abs(wrist_left.x - wrist_right.x) + abs(wrist_left.y - wrist_right.y)
        
        if dist_wrist < 0.2:
            if dedos_extremos(landmarks_left) and dedos_extremos(landmarks_right):
                return True
    return False

# Función para verificar si la palma está hacia abajo
def hacer(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist_y = landmarks[0].y
    knuckles_y = sum([landmarks[i].y for i in [5, 9, 13, 17]]) / 4
    return wrist_y < knuckles_y

# Función para verificar si los nudillos están por encima de la muñeca para validar casa y correr
def nudillos_sobre_muneca(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist_y = landmarks[mp_holistic.HandLandmark.WRIST].y
    knuckles_y = sum([landmarks[i].y for i in [
        mp_holistic.HandLandmark.INDEX_FINGER_MCP,
        mp_holistic.HandLandmark.MIDDLE_FINGER_MCP,
        mp_holistic.HandLandmark.RING_FINGER_MCP,
        mp_holistic.HandLandmark.PINKY_MCP]]) / 4
    return knuckles_y < wrist_y

# Función para detectar el gesto "Hola" correctamente
def detectar_hola(results):
    def is_palm_open(hand_landmarks):
        if not hand_landmarks:
            return False
        landmarks = hand_landmarks.landmark
        # Verificar que los dedos estén completamente extendidos, indicando una palma abierta
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y < landmarks[14].y and
                landmarks[20].y < landmarks[18].y)

    # Detectar si la palma izquierda o derecha está abierta
    left_hand_open = is_palm_open(results.left_hand_landmarks)
    right_hand_open = is_palm_open(results.right_hand_landmarks)

    # Solo devolver "Hola" si alguna de las palmas está completamente abierta
    if left_hand_open or right_hand_open:
        return "Hola"
    return None

# Función para detectar el gesto "Correr" correctamente
def detectar_correr(results):
    if not results.left_hand_landmarks or not results.right_hand_landmarks:
        return None

    left_landmarks = results.left_hand_landmarks.landmark
    right_landmarks = results.right_hand_landmarks.landmark
    
    # Verificar que los puños estén completamente cerrados
    left_puño = nudillos_sobre_muneca(results.left_hand_landmarks)
    right_puño = nudillos_sobre_muneca(results.right_hand_landmarks)
    
    # Solo detectar correr si ambos puños están cerrados
    if left_puño and right_puño:
        # Comprobar la posición de la muñeca y los nudillos
        left_wrist = left_landmarks[mp_holistic.HandLandmark.WRIST]
        left_middle_knuckle = left_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]

        right_wrist = right_landmarks[mp_holistic.HandLandmark.WRIST]
        right_middle_knuckle = right_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]

        # Verificar que las muñecas estén más altas que los nudillos
        if left_wrist.z > left_middle_knuckle.z and right_wrist.z > right_middle_knuckle.z:
            return "Correr"
    return None

# Función para detectar el gesto "Casa"
def detectar_gesto_casa(results):
    if results.left_hand_landmarks and results.right_hand_landmarks:
        left_hand = results.left_hand_landmarks.landmark
        right_hand = results.right_hand_landmarks.landmark

        left_index = (left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                      left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
        right_index = (right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                       right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)

        distance = ((left_index[0] - right_index[0])**2 +
                    (left_index[1] - right_index[1])**2)**0.5

        if distance < 0.1:
            if nudillos_sobre_muneca(results.left_hand_landmarks) and nudillos_sobre_muneca(results.right_hand_landmarks):
                return "Casa"
    return None

# Inicia la captura de la cámara
cap = cv2.VideoCapture(0)

# Usa el modelo Holistic de MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear la imagen horizontalmente
        frame = cv2.flip(frame, 1)
        
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realiza la detección
        results = holistic.process(image_rgb)

        # Dibuja los landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Detecta gestos
        mensaje = "No detectado"
        
        mensaje_hola = detectar_hola(results)
        if mensaje_hola:
            mensaje = mensaje_hola

        mensaje_correr = detectar_correr(results)
        if mensaje_correr:
            mensaje = mensaje_correr

        gesto_casa = detectar_gesto_casa(results)
        if gesto_casa:
            mensaje = gesto_casa

        if detectar_amigo(results.left_hand_landmarks, results.right_hand_landmarks):
            mensaje = "Amigo"
        
        if results.left_hand_landmarks and results.right_hand_landmarks:
            if hacer(results.left_hand_landmarks) and hacer(results.right_hand_landmarks):
                mensaje = "HACER"
        
        # Detecta la seña de "Universidad"
        if results.left_hand_landmarks:
            if detect_university_sign(results.left_hand_landmarks.landmark):
                mensaje = "Universidad"
        
        if results.right_hand_landmarks:
            if detect_university_sign(results.right_hand_landmarks.landmark):
                mensaje = "Universidad"

        # Muestra el mensaje en la pantalla
        cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Muestra el video
        cv2.imshow('Gestos Detectados', frame)

        # Sale con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
