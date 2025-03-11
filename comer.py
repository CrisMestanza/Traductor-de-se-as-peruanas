
# Función para detectar "Comer"
import cv2
import mediapipe as mp
import math
# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calcular_distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def detect_comer(landmarks, boca):
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]
    return calcular_distancia(index_finger, boca) < 0.1 or calcular_distancia(pinky_finger, boca) < 0.1
