
import cv2
import time
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling

# ---------------- CONFIGURATION ----------------
MODEL_PATH = Path("models/best_eye_model.keras")
IMG_SIZE = (93, 93)

CONFIDENCE_THRESHOLD = 0.5
WARNING_SECONDS = 2.0

# ---------------- LOAD MODEL ----------------
print("Loading eye-state model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

model = load_model(str(MODEL_PATH))
print("Model loaded successfully!")

# Check whether the model already normalizes pixel values
has_rescaling = any(
    isinstance(layer, Rescaling)
    for layer in model.layers
)

print("Model has Rescaling layer:", has_rescaling)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# ---------------- LOAD DETECTORS ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face detector load aagala.")

if eye_cascade.empty():
    raise RuntimeError("Eye detector load aagala.")

# ---------------- OPEN CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        "Webcam open aagala. Camera permission check pannunga."
    )

# Improve webcam resolution if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

closed_start = None

print("Webcam started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame read panna mudiyala.")
            break

        # Mirror preview
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        eye_states = []

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 180, 0),
                2
            )

            face_gray = gray[y:y + h, x:x + w]

            # Search only the upper part of the face
            upper_height = int(h * 0.60)
            upper_face = face_gray[:upper_height, :]

            detected_eyes = eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(20, 12)
            )

            # Larger detections first; suppress duplicates
            detected_eyes = sorted(
                detected_eyes,
                key=lambda item: item[2] * item[3],
                reverse=True
            )

            selected_eyes = []

            for eye in detected_eyes:
                ex, ey, ew, eh = eye
                duplicate = False

                for sx, sy, sw, sh in selected_eyes:
                    overlap_x = max(
                        0,
                        min(ex + ew, sx + sw) - max(ex, sx)
                    )
                    overlap_y = max(
                        0,
                        min(ey + eh, sy + sh) - max(ey, sy)
                    )

                    overlap_area = overlap_x * overlap_y
                    eye_area = ew * eh

                    if eye_area > 0 and overlap_area / eye_area > 0.4:
                        duplicate = True
                        break

                if not duplicate:
                    selected_eyes.append(
                        (ex, ey, ew, eh)
                    )

            # Process detected eyes
            for (ex, ey, ew, eh) in selected_eyes:
                eye_img = face_gray[
                    ey:ey + eh,
                    ex:ex + ew
                ]

                if eye_img.size == 0:
                    continue

                # Grayscale image, 93x93
                eye_img = cv2.resize(
                    eye_img,
                    IMG_SIZE
                )

                eye_img = eye_img.astype(np.float32)

                # Normalize only if model has no Rescaling layer
                if not has_rescaling:
                    eye_img = eye_img / 255.0

                # Shape: (1, 93, 93, 1)
                eye_img = np.expand_dims(
                    eye_img,
                    axis=(0, -1)
                )

                # Fast inference
                prediction = model(
                    eye_img,
                    training=False
                ).numpy().reshape(-1)

                if prediction.size == 0:
                    continue

                open_probability = float(prediction[0])

                # Model output is assumed to be sigmoid probability
                if open_probability >= CONFIDENCE_THRESHOLD:
                    state = "OPEN"
                    confidence = open_probability
                    color = (0, 255, 0)
                else:
                    state = "CLOSED"
                    confidence = 1.0 - open_probability
                    color = (0, 0, 255)

                eye_states.append(state)

                # Eye rectangle in full-frame coordinates
                fx1 = x + ex
                fy1 = y + ey
                fx2 = fx1 + ew
                fy2 = fy1 + eh

                cv2.rectangle(
                    frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{state} {confidence:.2f}",
                    (fx1, max(20, fy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        # ---------------- DROWSINESS STATUS ----------------
        now = time.time()

        if eye_states:
            # Warn only if every detected eye is classified closed
            if all(state == "CLOSED" for state in eye_states):
                if closed_start is None:
                    closed_start = now

                closed_duration = now - closed_start

                if closed_duration >= WARNING_SECONDS:
                    message = "WARNING: EYES CLOSED!"
                    message_color = (0, 0, 255)
                else:
                    message = "Eyes closed - checking..."
                    message_color = (0, 165, 255)

            else:
                closed_start = None
                message = "Eyes open"
                message_color = (0, 255, 0)

        else:
            # No detection does not mean eyes are closed
            closed_start = None
            message = "Eyes not detected"
            message_color = (0, 255, 255)

        cv2.putText(
            frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            message_color,
            2
        )

        cv2.imshow(
            "Eye State Detection - Press Q to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
