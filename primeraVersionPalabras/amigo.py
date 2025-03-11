import cv2
import mediapipe as mp

# Inicializa MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para verificar si los dedos están extendidos (mano abierta)
def dedos_extremos(landmarks):
    """
    Función que verifica si los dedos están extendidos.
    Esta es una forma de determinar si la mano está abierta.
    """
    # Verifica si los dedos están extendidos comparando la posición de los dedos
    # Estirados (x, y) de los puntos de los dedos con los puntos cercanos a la muñeca
    thumb_tip = landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

    # Lógica simple: si la distancia entre los puntos de los dedos y la muñeca es suficientemente grande
    # entonces los dedos están extendidos (esto es solo un ejemplo simple)
    if (thumb_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        index_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        middle_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        ring_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y and
        pinky_tip.y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y):
        return True
    return False

# Función para detectar el gesto de "amigo"
def detectar_amigo(landmarks_left, landmarks_right):
    """
    Detecta el gesto de 'amigo' basado en la relación de las posiciones de las muñecas y los dedos.
    """
    if landmarks_left and landmarks_right:
        # Extrae las coordenadas de las muñecas
        wrist_left = landmarks_left.landmark[mp_holistic.HandLandmark.WRIST]
        wrist_right = landmarks_right.landmark[mp_holistic.HandLandmark.WRIST]

        # Calcular la distancia entre las muñecas (para verificar si están cerca)
        dist_wrist = abs(wrist_left.x - wrist_right.x) + abs(wrist_left.y - wrist_right.y)
        
        # Verifica si las muñecas están cerca (ajustar según sea necesario)
        if dist_wrist < 0.2:
            # Verifica si ambos dedos están extendidos
            if dedos_extremos(landmarks_left) and dedos_extremos(landmarks_right):
                return True

    return False

# Inicia la captura de la cámara
cap = cv2.VideoCapture(0)

# Usa el modelo Holistic de MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convierte la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realiza la detección
        results = holistic.process(image_rgb)

        # Dibuja los landmarks en la imagen
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Verifica si el gesto de "amigo" está presente
        if detectar_amigo(results.left_hand_landmarks, results.right_hand_landmarks):
            cv2.putText(frame, 'Amigo', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Muestra la imagen procesada
        cv2.imshow("Detección de Gesto de Amigo", frame)

        # Sale si presionas 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera la captura y cierra la ventana
cap.release()
cv2.destroyAllWindows()
