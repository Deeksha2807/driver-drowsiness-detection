import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "models/best_eye_model.keras"
IMG_SIZE = (93, 93)

WARNING_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0

# ---------------- LOAD MODEL ----------------
print("Loading eye-state model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# ---------------- LOAD CASCADES ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Haar cascade load aagala. OpenCV installation check pannu.")

# ---------------- EYE PREDICTION ----------------
def predict_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None, 0.0

    eye_img = cv2.resize(eye_img, IMG_SIZE)
    eye_img = eye_img.astype("float32")

    eye_img = np.expand_dims(eye_img, axis=-1)
    eye_img = np.expand_dims(eye_img, axis=0)

    # model.predict() badhil direct inference
    prediction = model(eye_img, training=False)
    open_probability = float(np.asarray(prediction).reshape(-1)[0])

    if open_probability >= CONFIDENCE_THRESHOLD:
        return "OPEN", open_probability
    else:
        return "CLOSED", open_probability


# ---------------- OPEN CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera open aagala. CAMERA_INDEX = 1 try pannalaam.")
    raise SystemExit

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam started. Quit panna Q press pannunga.")

closed_start_time = None
warning_active = False

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera frame read aagala.")
            time.sleep(0.1)
            continue
        

        # Mirror view: selfie maadhiri display aagum
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        eyes_found = 0
        closed_eyes = 0
        eye_results = []

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (255, 180, 0), 2
            )

            # Face ROI
            face_gray = gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]

            # Eyes generally upper half of face
            upper_gray = face_gray[0:int(h * 0.65), :]
            upper_color = face_color[0:int(h * 0.65), :]

            eyes = eye_cascade.detectMultiScale(
                upper_gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(15, 10)
            )

            # Avoid classifying too many false eye detections
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

            for (ex, ey, ew, eh) in eyes:
                eye_img = upper_gray[ey:ey + eh, ex:ex + ew]

                state, open_prob = predict_eye(eye_img)

                if state is None:
                    continue

                eyes_found += 1

                # Draw eye box and prediction
                color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    color, 2
                )

                cv2.putText(
                    frame,
                    f"{state} {open_prob:.2f}",
                    (x + ex, max(20, y + ey - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                eye_results.append(state)

                if state == "CLOSED":
                    closed_eyes += 1

        # ---------------- DROWSINESS LOGIC ----------------
        # Warning only when both eyes are detected and closed
        both_eyes_closed = (
            eyes_found >= 2 and closed_eyes >= 2
        )

        if both_eyes_closed:
            if closed_start_time is None:
                closed_start_time = time.time()

            elapsed = time.time() - closed_start_time

            if elapsed >= WARNING_SECONDS:
                warning_active = True
        else:
            closed_start_time = None
            warning_active = False

        # ---------------- DISPLAY STATUS ----------------
        if len(faces) == 0:
            status_text = "Face not detected"
            status_color = (0, 255, 255)
        elif eyes_found == 0:
            status_text = "Eyes not detected"
            status_color = (0, 255, 255)
        elif warning_active:
            status_text = "DROWSINESS ALERT!"
            status_color = (0, 0, 255)
        else:
            status_text = "Driver Alert"
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2
        )

        if closed_start_time is not None and not warning_active:
            remaining = max(
                0, WARNING_SECONDS - (time.time() - closed_start_time)
            )
            cv2.putText(
                frame,
                f"Eyes closed: {remaining:.1f}s",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        if warning_active:
            cv2.putText(
                frame,
                "WAKE UP!",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3
            )

            # Visual alert only; no sound
            cv2.rectangle(
                frame,
                (5, 5),
                (frame.shape[1] - 5, frame.shape[0] - 5),
                (0, 0, 255),
                4
            )

        # IMPORTANT: show the original camera frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
