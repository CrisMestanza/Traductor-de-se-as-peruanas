import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Función para verificar si la parte trasera del puño apunta hacia la persona
def is_back_fist(hand_landmarks):
    landmarks = hand_landmarks.landmark

    # Coordenadas clave
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_knuckle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Verificar que la muñeca esté detrás de los nudillos
    if wrist.z > middle_knuckle.z:  # Eje z: muñeca detrás
        return True
    return False

# Captura de video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)

        # Convertir de vuelta a BGR para mostrar
        image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detectar manos
        if resultados.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(resultados.multi_hand_landmarks, resultados.multi_handedness):
                label = handedness.classification[0].label  # "Left" o "Right"
                back_fist_detected = is_back_fist(hand_landmarks)

                # Si se detecta un puño hacia atrás, mostrar "Correr"
                if back_fist_detected:
                    cv2.putText(image, "Correr", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Dibujar las manos detectadas
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar el resultado
        cv2.imshow('Detección de Manos', image)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
