import argparse
import time
import cv2
import mediapipe as mp

from pythonosc.udp_client import SimpleUDPClient

# =========================================================
# OSC -> PURE DATA
# =========================================================

# IP local + puerto de Pure Data
client = SimpleUDPClient("127.0.0.1", 9000)

# =========================================================
# UMBRALES --- AJUSTAR LUEGO
# =========================================================

BLINK_THRESHOLD = 0.5
BROWS_THRESHOLD = 0.7

# =========================================================
# ARGUMENTOS
# =========================================================

parser = argparse.ArgumentParser(
    description="Detección de gestos faciales con MediaPipe + OSC."
)

parser.add_argument(
    "--show-landmarks",
    action="store_true",
    help="Dibuja los landmarks faciales.",
)

args = parser.parse_args()

# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="model/face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1,
)

# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando detector...")
print("Presiona 'q' para salir.")

# =========================================================
# ESTADOS DISCRETOS
# =========================================================

blink_left_state = 0
blink_right_state = 0
brows_up_state = 0


# =========================================================
# LOOP PRINCIPAL
# =========================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Error: no se puede recibir video.")
            break

        # Efecto espejo
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp requerido por VIDEO mode
        timestamp_ms = int(time.time() * 1000)

        # Inferencia
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # =================================================
        # TEXTOS
        # =================================================

        jaw_text = "Jaw: ---"
        left_text = "Left Blink: ---"
        right_text = "Right Blink: ---"
        brow_text = "Brows Up: ---"

        # =================================================
        # DETECCIÓN
        # =================================================

        if result.face_blendshapes:

            blendshapes = result.face_blendshapes[0]

            # =============================================
            # BUSCAR GESTOS
            # =============================================

            jaw_open = next(
                (
                    b for b in blendshapes
                    if b.category_name == "jawOpen"
                ),
                None
            )

            left_blink = next(
                (
                    b for b in blendshapes
                    if b.category_name == "eyeBlinkLeft"
                ),
                None
            )

            right_blink = next(
                (
                    b for b in blendshapes
                    if b.category_name == "eyeBlinkRight"
                ),
                None
            )

            # CEJAS 

        brows_up = next(
            (
                b for b in blendshapes
                if b.category_name == "browOuterUpRight"
                    or b.category_name == "browOuterUpLeft" 
                    or b.category_name == "browInnerUp"

                # Se fija si se levanta alguna ceja
            ),
            None
            )
            
            # =============================================
            # BROWS UP ----> DISCRETO
            # =============================================

        if brows_up:

            brows_score = brows_up.score

            new_brows_up_state = int(brows_score > BROWS_THRESHOLD)

   
            if(
                brows_up_state == 0 and
                new_brows_up_state == 1):

                client.send_message(
                "/browsUp",
                1
        )

            brows_up_state = new_brows_up_state

            brow_text = (
            f"Brows Up: {brows_up_state}"
    )

                
            # =============================================
            # JAW OPEN ----> CONTINUO
            # =============================================

            if jaw_open:

                jaw_score = jaw_open.score

                # Enviar OSC
                client.send_message(
                    "/jawOpen",
                    jaw_score
                )

                if jaw_score > 0.30:
                    jaw_text = f"Jaw OPEN ({jaw_score:.2f})"
                else:
                    jaw_text = f"Jaw closed ({jaw_score:.2f})"

            # =============================================
            # LEFT BLINK ----> DISCRETO
            # =============================================

            if left_blink:

                left_score = left_blink.score
                new_left_state = int(left_score > BLINK_THRESHOLD)
                

            if new_left_state != blink_left_state:

                if (blink_left_state == 0 and new_left_state == 1):
                    client.send_message("/blinkLeft",1)

            blink_left_state = new_left_state

            if blink_left_state != 0:
                client.send_message(
                "/blinkLeft",
                blink_left_state
            )

            left_text = (
            f"Left Blink: {blink_left_state}"
            )

            # =============================================
            # RIGHT BLINK ----> DISCRETO
            # =============================================

            if right_blink:

                right_score = right_blink.score
                new_right_state = int(right_score > BLINK_THRESHOLD)
               

            if new_right_state != blink_right_state:

                if (blink_right_state == 0 and
                    new_right_state == 1):
                    client.send_message("/blinkRight",1)

            blink_right_state = new_right_state

            if blink_right_state != 0:
                client.send_message(
                "/blinkRight",
                blink_right_state
            )

            right_text = (
            f"Right Blink: {blink_right_state}"
            )
            # =============================================
            # LANDMARKS
            # =============================================

            if args.show_landmarks and result.face_landmarks:

                for face_landmarks in result.face_landmarks:

                    for landmark in face_landmarks:

                        cx = int(landmark.x * w)
                        cy = int(landmark.y * h)

                        cv2.circle(
                            frame,
                            (cx, cy),
                            1,
                            (0, 255, 0),
                            -1
                        )

        # =================================================
        # INTERFAZ USUARIO
        # =================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 180),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            jaw_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            left_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            right_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            brow_text,
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        ),

        # Mostrar frame
        cv2.imshow(
            "Face Gesture Detector + OSC",
            frame
        )

        # Salir con q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# =========================================================
# LIMPIEZA
# =========================================================

cap.release()
cv2.destroyAllWindows()

print("Script finalizado.")