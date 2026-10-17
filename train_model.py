"""
Train a CNN-LSTM model on labeled video-frame sequences.

Expected folder structure:
data/
  train/
    alert/
      video_001/frame_000001.jpg
      video_001/frame_000002.jpg
      video_002/...
    drowsy/
      video_003/...
  val/
    alert/video_010/...
    drowsy/video_011/...
  test/
    alert/video_020/...
    drowsy/video_021/...

Important: keep frames from the same source video in only one split.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 96
SEQ_LEN = 16
BATCH_SIZE = 4
CLASS_NAMES = ["alert", "drowsy"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_sequences(split_dir, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    xs, ys = [], []
    split_dir = Path(split_dir)

    for label_id, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"Warning: missing directory: {class_dir}")
            continue

        video_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        # Also accept images directly inside class folder as one sequence source.
        direct_images = sorted(
            [p for p in class_dir.iterdir()
             if p.is_file() and p.suffix.lower() in VALID_EXTS]
        )
        if direct_images:
            video_dirs.append(class_dir)

        for video_dir in sorted(video_dirs):
            image_paths = sorted(
                [p for p in video_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in VALID_EXTS]
            )
            if len(image_paths) < seq_len:
                continue

            # Non-overlapping chunks reduce duplicate near-identical sequences.
            for start in range(0, len(image_paths) - seq_len + 1, seq_len):
                chunk = image_paths[start:start + seq_len]
                frames = []
                valid = True
                for path in chunk:
                    image = cv2_read(path, img_size)
                    if image is None:
                        valid = False
                        break
                    frames.append(image)
                if valid:
                    xs.append(np.asarray(frames, dtype=np.float32) / 255.0)
                    ys.append(label_id)

    if not xs:
        return np.empty((0, seq_len, img_size, img_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def cv2_read(path, img_size):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        print(f"Skipping unreadable image: {path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    return image


def build_model(seq_len=SEQ_LEN, img_size=IMG_SIZE):
    inputs = tf.keras.Input(shape=(seq_len, img_size, img_size, 3))
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")
    )(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")
    )(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D())(x)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
    )(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    x = tf.keras.layers.LSTM(64, dropout=0.25)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="Dataset root folder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--output", default="drowsiness_cnn_lstm.keras")
    args = parser.parse_args()

    train_x, train_y = load_sequences(Path(args.data) / "train", args.seq_len, args.img_size)
    val_x, val_y = load_sequences(Path(args.data) / "val", args.seq_len, args.img_size)
    test_x, test_y = load_sequences(Path(args.data) / "test", args.seq_len, args.img_size)

    for name, x, y in [("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)]:
        print(f"{name}: {len(x)} sequences")
        if len(y):
            print("  class counts (alert=0, drowsy=1):", np.bincount(y, minlength=2))

    if len(train_x) == 0:
        raise SystemExit(
            "No training sequences found. Check the folder structure and make sure each video "
            f"has at least {args.seq_len} images."
        )
    if len(np.unique(train_y)) < 2:
        raise SystemExit("Training data must contain both alert and drowsy sequences.")
    if len(val_x) == 0:
        raise SystemExit("Validation sequences not found. Add labeled videos under data/val/.")

    model = build_model(args.seq_len, args.img_size)
    model.summary()

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_y), y=train_y)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(train_y), weights)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5)
    ]
    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )
    model.save(args.output)
    print(f"Saved model to {args.output}")

    if len(test_x):
        loss, acc, precision, recall = model.evaluate(test_x, test_y, verbose=0)
        probabilities = model.predict(test_x, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)
        print("\nTest results")
        print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print("\nClassification report:")
        print(classification_report(test_y, predictions, target_names=CLASS_NAMES, zero_division=0))
        print("Confusion matrix (rows=true, columns=predicted; alert,drowsy):")
        print(confusion_matrix(test_y, predictions))
    else:
        print("No test sequences found; skipped final evaluation.")


if __name__ == "__main__":
    import cv2
    main()
