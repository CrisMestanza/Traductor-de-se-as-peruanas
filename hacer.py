# Función para verificar si la palma está hacia abajo
import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def hacer(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist_y = landmarks[0].y
    knuckles_y = sum([landmarks[i].y for i in [5, 9, 13, 17]]) / 4
    return wrist_y < knuckles_y