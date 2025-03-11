# Función para detectar "Casa"
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

def detectar_gesto_casa(results):
    if results.left_hand_landmarks and results.right_hand_landmarks:
        left_hand = results.left_hand_landmarks.landmark
        right_hand = results.right_hand_landmarks.landmark

        left_index = (left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                      left_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
        right_index = (right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x,
                       right_hand[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)

        distance = math.sqrt((left_index[0] - right_index[0])**2 + (left_index[1] - right_index[1])**2)

        if distance < 0.1 and nudillos_sobre_muneca(results.left_hand_landmarks) and nudillos_sobre_muneca(results.right_hand_landmarks):
            return "Casa"
    return None