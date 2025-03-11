import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para calcular la distancia entre dos puntos
def calcular_distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Función para detectar la señal de "comer"
def detect_comer(landmarks, boca):
    """
    Función que detecta si los dedos índice o meñique están cerca de la boca.
    """
    # Definir los puntos de los dedos: índice (punto 8) y meñique (punto 20)
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]

    # Calcular la distancia entre los dedos y la boca
    distancia_indice = calcular_distancia(index_finger, boca)
    distancia_menique = calcular_distancia(pinky_finger, boca)

    # Si la distancia entre los dedos y la boca es pequeña, se considera "comer"
    if distancia_indice < 0.1 or distancia_menique < 0.1:
        return True
    return False

# Configurar el modelo Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear la imagen horizontalmente para mayor facilidad
        frame = cv2.flip(frame, 1)
        
        # Convertir la imagen a RGB para que MediaPipe la procese
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen y detectar las señales
        results = holistic.process(rgb_frame)

        # Si se detecta la boca
        if results.face_landmarks:
            boca = results.face_landmarks.landmark[13]  # Centro de la boca

            # Si se detecta la mano izquierda o derecha
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # Si se detecta la seña de "comer" en la mano izquierda
                if results.left_hand_landmarks:
                    if detect_comer(results.left_hand_landmarks.landmark, boca):
                        cv2.putText(frame, 'Comiendo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Si se detecta la seña de "comer" en la mano derecha
                if results.right_hand_landmarks:
                    if detect_comer(results.right_hand_landmarks.landmark, boca):
                        cv2.putText(frame, 'Comiendo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Dibujar las detecciones de la pose y las manos
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Mostrar la imagen con las detecciones
        cv2.imshow('Comiendo Sign', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
