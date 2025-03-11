
# Función para detectar la seña de 'universidad'
import cv2
import mediapipe as mp
import math
# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def detect_university_sign(landmarks):
    index_finger = landmarks[8]
    pinky_finger = landmarks[20]

    if (index_finger.y < landmarks[7].y and pinky_finger.y < landmarks[19].y):
        thumb_finger = landmarks[4]
        middle_finger = landmarks[12]
        ring_finger = landmarks[16]

        if (thumb_finger.y > landmarks[3].y and middle_finger.y > landmarks[11].y and ring_finger.y > landmarks[15].y):
            return True
    return False