import cv2
import mediapipe as mp
import pyttsx3
import threading  # Importar módulo para manejar hilos
import math
from universidad import detect_university_sign
from amigo import detectar_amigo
from hola import detectar_hola
from correr import detectar_correr
from casa import detectar_gesto_casa
from comer import detect_comer
from hacer import hacer

# Holistic jejeje
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Voz
engine = pyttsx3.init()
engine.setProperty('rate', 120)  

# Hilo jejeje
def hablar_en_hilo(mensaje):
    hilo = threading.Thread(target=hablar, args=(mensaje,))
    hilo.start()

def hablar(mensaje):
    engine.say(mensaje)
    engine.runAndWait()

# Camara
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

        cv2.putText(frame, mensaje, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Emitir el audio en un hilo para evitar bloqueos
        if mensaje != "No detectado":
            hablar_en_hilo(mensaje)

        cv2.imshow('Gestos Detectados', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
