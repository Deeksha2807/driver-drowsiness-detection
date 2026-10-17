"""Live webcam inference for the trained CNN-LSTM model."""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
SEQ_LEN = 16


def beep():
    try:
        import winsound
        winsound.Beep(1200, 450)
    except Exception:
        print("\a", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="drowsiness_cnn_lstm.keras")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Probability threshold for drowsy prediction")
    parser.add_argument("--consecutive", type=int, default=3,
                        help="Number of consecutive drowsy predictions before warning")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or try --camera 1.")

    frames = deque(maxlen=SEQ_LEN)
    drowsy_count = 0
    last_beep = 0.0
    print("Press Q to quit. Keep the face similarly framed to the training images.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break

        # The same whole-frame preprocessing is used in training.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frames.append(small)

        status = "Collecting frames..."
        color = (0, 220, 255)
        probability = None

        if len(frames) == SEQ_LEN:
            sequence = np.expand_dims(np.asarray(frames, dtype=np.float32), axis=0)
            probability = float(model.predict(sequence, verbose=0)[0][0])
            if probability >= args.threshold:
                drowsy_count += 1
            else:
                drowsy_count = 0

            if drowsy_count >= args.consecutive:
                status = f"DROWSINESS WARNING ({probability:.2f})"
                color = (0, 0, 255)
                now = time.time()
                if now - last_beep > 2.0:
                    beep()
                    last_beep = now
            else:
                status = f"Alert / monitoring (drowsy score: {probability:.2f})"
                color = (0, 200, 0)

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Frames: {len(frames)}/{SEQ_LEN}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Driver Drowsiness Detection - Prototype", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
