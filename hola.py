# Función para detectar "Hola"
import cv2
import mediapipe as mp
import math
# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calcular_distancia(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def dedos_separados(hand_landmarks):
    landmarks = hand_landmarks.landmark
    dist_index_middle = calcular_distancia(landmarks[8], landmarks[12])
    dist_middle_ring = calcular_distancia(landmarks[12], landmarks[16])
    dist_ring_pinky = calcular_distancia(landmarks[16], landmarks[20])

    # Verificar que las distancias sean significativas para considerar los dedos separados
    return dist_index_middle > 0.05 and dist_middle_ring > 0.05 and dist_ring_pinky > 0.05


def detectar_hola(results):
    def is_palm_open(hand_landmarks):
        if not hand_landmarks:
            return False
        
        landmarks = hand_landmarks.landmark
        
        # Verificar si los dedos están extendidos (yema por debajo de la falange media)
        dedos_extendidos = all(
            landmarks[finger_tip].y < landmarks[finger_dip].y
            for finger_tip, finger_dip in [
                (8, 6),   # Pulgar (yema vs falange media)
                (12, 10), # Índice (yema vs falange media)
                (16, 14), # Medio (yema vs falange media)
                (20, 18)  # Anular (yema vs falange media)
            ]
        )
        
        # Verificar que la palma está abierta y que los dedos están separados
        # Se asume que si los dedos están extendidos y separados, la palma está abierta
        palma_abierta = dedos_extendidos and dedos_separados(hand_landmarks)
        
        return palma_abierta

    # Detectar si la palma abierta y dedos parados están en la mano izquierda o derecha
    if is_palm_open(results.left_hand_landmarks) or is_palm_open(results.right_hand_landmarks):
        return "Hola"
    
    return None