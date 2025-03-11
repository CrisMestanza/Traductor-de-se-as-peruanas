
# Función para detectar "Correr"
import cv2
import mediapipe as mp
import math
# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def detectar_correr(results):
    if results.left_hand_landmarks and results.right_hand_landmarks:
        if nudillos_sobre_muneca(results.left_hand_landmarks) and nudillos_sobre_muneca(results.right_hand_landmarks):
            return "Correr"
    return None