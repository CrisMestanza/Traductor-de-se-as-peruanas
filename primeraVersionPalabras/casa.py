import cv2
import mediapipe as mp

# Función para detectar las manos y mostrar los mensajes en pantalla
def detectar_manos():
    # Inicializa MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Configuración de la cámara
    cap = cv2.VideoCapture(0)

    # Inicia el modelo Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                continue

            # Convierte la imagen a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Dibuja los resultados (huesos de la mano)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Inicializa las variables de estado
            mensaje = ""

            # Comprobar si ambos dedos pulgar y meñique están alzados
            if results.left_hand_landmarks and results.right_hand_landmarks:
                # Puntos clave para pulgar y meñique (índices de los puntos según MediaPipe)
                left_thumb = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
                left_pinky = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
                right_thumb = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
                right_pinky = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

                # Compara la posición de los dedos (si el Y del pulgar es menor que el del meñique, significa que están alzados)
                if (left_thumb.y < left_pinky.y and right_thumb.y < right_pinky.y):
                    mensaje = "Pulgar y meñique alzados"

                # Verifica la orientación de las manos
                left_wrist = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                right_wrist = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]

                # Si las muñecas están hacia atrás en el eje X, podrías determinar si las manos están hacia atrás
                if left_wrist.x < 0.5 and right_wrist.x < 0.5:
                    mensaje = "Manos hacia atrás"

            # Muestra el mensaje en la imagen
            cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Muestra la imagen con los resultados y el mensaje
            cv2.imshow('MediaPipe Holistic', frame)

            # Salir si presionas la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cierra la cámara
    cap.release()
    cv2.destroyAllWindows()

# Llamada a la función
detectar_manos()
