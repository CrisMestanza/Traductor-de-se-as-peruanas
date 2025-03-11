
import cv2
import mediapipe as mp
import math
# Función para detectar "Amigo"

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def dedos_extremos(landmarks):
    return all(
        landmarks.landmark[finger_tip].y < landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
        for finger_tip in [
            mp_holistic.HandLandmark.THUMB_TIP,
            mp_holistic.HandLandmark.INDEX_FINGER_TIP,
            mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
            mp_holistic.HandLandmark.RING_FINGER_TIP,
            mp_holistic.HandLandmark.PINKY_TIP,
        ]
    )

def detectar_amigo(landmarks_left, landmarks_right):
    if landmarks_left and landmarks_right:
        wrist_left = landmarks_left.landmark[mp_holistic.HandLandmark.WRIST]
        wrist_right = landmarks_right.landmark[mp_holistic.HandLandmark.WRIST]

        dist_wrist = abs(wrist_left.x - wrist_right.x) + abs(wrist_left.y - wrist_right.y)

        if dist_wrist < 0.2 and dedos_extremos(landmarks_left) and dedos_extremos(landmarks_right):
            return "Amigo"
    return None