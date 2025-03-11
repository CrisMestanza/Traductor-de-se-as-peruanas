import cv2
import mediapipe as mp

# Inicializar Mediapipe Holistic
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Función para determinar si la palma está abierta o cerrada
def detectar_mano(resultados):
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            # Coordenadas de los nudillos y la punta del dedo medio
            medioMcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            medioTip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            
            #Dedo index
            indexMcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            indexTip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            
            #Dedo anular
            anularMcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            anularTip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
             
            #Dedo meñique
            meniqueMcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
            meniqueTip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            
            # Si la punta del dedo medio está más abajo que el nudillo, es un puño
            if (medioTip < medioMcp) and (indexTip < indexMcp) and (anularTip < anularMcp) and (meniqueTip < meniqueMcp):
                return "Hola"  # Palma cerrada
            
    return None

# Captura de video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB para Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)

    # Dibujar la mano y obtener el estado
    mensaje = "No detectado"
    if resultados.multi_hand_landmarks is not None and len(resultados.multi_hand_landmarks) == 1:
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mensaje = detectar_mano(resultados)

    # Mostrar el mensaje en la pantalla
    cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video
    cv2.imshow('Holistic Hola-Chau', frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
