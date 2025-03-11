import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para calcular la distancia entre dos puntos
def calcular_distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

# Función para detectar "Comer" con una mejor lógica
def detect_comer(landmarks, boca):
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]

    # Verifica si el dedo índice o meñique están cerca de la boca y la mano está inclinada correctamente
    if calcular_distancia(index_finger, boca) < 0.1 or calcular_distancia(pinky_finger, boca) < 0.1:
        return True
    return False

# Función para verificar si la palma está abierta
def is_palm_open(hand_landmarks):
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    return all(
        landmarks[finger_tip].y < landmarks[finger_dip].y
        for finger_tip, finger_dip in [
            (8, 6), (12, 10), (16, 14), (20, 18)
        ]
    )

# Función para detectar "Hola"
def detectar_hola(results):
    if is_palm_open(results.left_hand_landmarks) or is_palm_open(results.right_hand_landmarks):
        return "Hola"
    return None

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

        # Detectar "Comer"
        if results.face_landmarks:
            boca = results.face_landmarks.landmark[13]  # Centro de la boca
            if results.left_hand_landmarks and detect_comer(results.left_hand_landmarks.landmark, boca):
                mensaje = "Comer"
            if results.right_hand_landmarks and detect_comer(results.right_hand_landmarks.landmark, boca):
                mensaje = "Comer"

        # Solo detecta "Hola" si no se ha detectado "Comer"
        if mensaje == "No detectado":
            mensaje = detectar_hola(results) or mensaje

        # Dibujar y mostrar el mensaje en la pantalla
        cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Gestos Detectados', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
