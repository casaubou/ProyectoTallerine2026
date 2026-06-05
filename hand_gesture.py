import argparse
import time
import cv2
import mediapipe as mp

# Hand skeleton connections (21 landmarks, 0-indexed)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),           # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),       # Medio
    (9, 13), (13, 14), (14, 15), (15, 16),     # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),    # Meñique
    (0, 17),                                   # Palma
]

# 1. Argumentos por línea de comandos
parser = argparse.ArgumentParser(
    description="Detección de gestos de mano en tiempo real usando MediaPipe GestureRecognizer."
)
parser.add_argument(
    "--show-landmarks",
    action="store_true",
    help="Si se activa, dibuja el esqueleto de la mano en el video.",
)
args = parser.parse_args()

# 2. Inicialización del GestureRecognizer (Tasks API)
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="model/gesture_recognizer.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)

# 3. Captura de video (Webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando detector de gestos de mano... Presiona 'q' para salir.")

# Gestos posibles: None | Closed_Fist | Open_Palm | Pointing_Up
#                  Thumb_Down | Thumb_Up | Victory | ILoveYou

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir video de la cámara.")
            break

        # Efecto espejo
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe requiere RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        gesture_text = "Gesto: Ninguno"
        probability_text = "Prob: 0.00"
        hand_text = "Mano: -"

        # 4. Procesar resultados
        if result.gestures:
            top_gesture = result.gestures[0][0]  # primera mano, clasificación top
            hand_label = (
                result.handedness[0][0].display_name if result.handedness else "-"
            )
            gesture_text = f"Gesto: {top_gesture.category_name}"
            probability_text = f"Prob: {top_gesture.score:.2f}"
            hand_text = f"Mano: {hand_label}"

            # 5. Dibujar esqueleto si el argumento fue activado
            if args.show_landmarks and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    # Puntos
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    # Conexiones
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        x0 = int(hand_landmarks[start_idx].x * w)
                        y0 = int(hand_landmarks[start_idx].y * h)
                        x1 = int(hand_landmarks[end_idx].x * w)
                        y1 = int(hand_landmarks[end_idx].y * h)
                        cv2.line(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)

        # 6. Overlay de información
        cv2.rectangle(frame, (10, 10), (420, 135), (0, 0, 0), -1)
        cv2.putText(
            frame, gesture_text, (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, probability_text, (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, hand_text, (20, 124),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA,
        )

        cv2.imshow("Detector de Gestos de Mano (MediaPipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Limpieza de recursos
cap.release()
cv2.destroyAllWindows()
print("Script finalizado correctamente.")
