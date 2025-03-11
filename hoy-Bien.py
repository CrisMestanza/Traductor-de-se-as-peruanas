import cv2
import mediapipe as mp
import math
from universidad import detect_university_sign
from amigo import detectar_amigo
from hola import detectar_hola
from correr import detectar_correr
from casa import detectar_gesto_casa
from comer import detect_comer
from hacer import hacer

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Función para detectar si los pulgares están alzados
def pulgares_alzados(hand_landmarks, thumb_tip_idx, thumb_base_idx):
    """ Verifica si los pulgares están alzados. """
    thumb_up = hand_landmarks.landmark[thumb_tip_idx].y < hand_landmarks.landmark[thumb_base_idx].y
    return thumb_up

# Función para detectar si los meñiques están alzados
def meniques_alzados(hand_landmarks, pinky_tip_idx, pinky_base_idx):
    """ Verifica si los meñiques están alzados. """
    pinky_up = hand_landmarks.landmark[pinky_tip_idx].y < hand_landmarks.landmark[pinky_base_idx].y
    return pinky_up

# Función para verificar la condición para predecir "Hoy"
def predecir_hoy(left_thumb_up, right_thumb_up, left_pinky_up, right_pinky_up):
    """ Verifica la condición para predecir "Hoy" """
    if left_thumb_up and right_thumb_up and left_pinky_up and right_pinky_up:
        return "Hoy"
    return ""

# Función para verificar la condición para predecir "Bien"
def predecir_bien(left_thumb_up, right_thumb_up):
    """ Verifica la condición para predecir "Bien" """
    if left_thumb_up and right_thumb_up:
        return "Bien"
    return ""

# Iniciar predicción con la cámara
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
        mensaje = detectar_amigo(
            results.left_hand_landmarks, results.right_hand_landmarks) or mensaje

        if results.left_hand_landmarks and results.right_hand_landmarks:
            if hacer(results.left_hand_landmarks) and hacer(results.right_hand_landmarks):
                mensaje = "Hacer"

        if results.left_hand_landmarks and detect_university_sign(results.left_hand_landmarks.landmark):
            mensaje = "Universidad"

        if results.right_hand_landmarks and detect_university_sign(results.right_hand_landmarks.landmark):
            mensaje = "Universidad"

        # Detectar "Hoy" y "Bien"
        if results.left_hand_landmarks and results.right_hand_landmarks:
            # Verifica si los pulgares están alzados en la mano izquierda
            left_thumb_up = pulgares_alzados(
                results.left_hand_landmarks,
                mp_holistic.HandLandmark.THUMB_TIP,
                mp_holistic.HandLandmark.THUMB_IP
            )
            # Verifica si los pulgares están alzados en la mano derecha
            right_thumb_up = pulgares_alzados(
                results.right_hand_landmarks,
                mp_holistic.HandLandmark.THUMB_TIP,
                mp_holistic.HandLandmark.THUMB_IP
            )

            # Verifica si los meñiques están alzados en la mano izquierda
            left_pinky_up = meniques_alzados(
                results.left_hand_landmarks,
                mp_holistic.HandLandmark.PINKY_TIP,
                mp_holistic.HandLandmark.PINKY_MCP
            )
            # Verifica si los meñiques están alzados en la mano derecha
            right_pinky_up = meniques_alzados(
                results.right_hand_landmarks,
                mp_holistic.HandLandmark.PINKY_TIP,
                mp_holistic.HandLandmark.PINKY_MCP
            )

            # Llamada a las funciones de predicción
            mensaje_hoy = predecir_hoy(left_thumb_up, right_thumb_up, left_pinky_up, right_pinky_up)
            if mensaje_hoy:
                mensaje = mensaje_hoy
            else:
                mensaje_bien = predecir_bien(left_thumb_up, right_thumb_up)
                if mensaje_bien:
                    mensaje = mensaje_bien

        cv2.putText(frame, mensaje, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Gestos Detectados', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()