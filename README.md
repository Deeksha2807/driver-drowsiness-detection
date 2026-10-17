# Driver Drowsiness Detection Using Sequential Facial Images

A student-project starter implementation using a CNN-LSTM model. The CNN extracts visual features from frames and the LSTM learns temporal patterns across a sequence. A webcam application uses the trained model to display a status and issue a warning.

> This is a research/educational prototype, not a certified vehicle safety system. Do not rely on it to decide whether it is safe to drive.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Webcam for live testing
- Labeled image sequences from videos
- Windows, Linux, or macOS

Install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If TensorFlow installation fails, use a compatible Python version and follow the official TensorFlow installation instructions for your operating system.

## 2. Dataset format

Each video should be stored in its own folder. Put extracted frames in chronological order and name them consistently, for example `frame_000001.jpg`, `frame_000002.jpg`.

```text
data/
  train/
    alert/
      video_001/
        frame_000001.jpg
        frame_000002.jpg
        ...
      video_002/
        ...
    drowsy/
      video_003/
        frame_000001.jpg
        ...
  val/
    alert/video_010/...
    drowsy/video_011/...
  test/
    alert/video_020/...
    drowsy/video_021/...
```

- `alert`: frames from a video labeled alert.
- `drowsy`: frames from a video labeled drowsy.
- Each video folder needs at least 16 usable images by default.
- The script makes non-overlapping sequences of 16 frames.
- The default image size is 96 × 96 pixels.
- Keep all frames from a particular source video in only one split. Prefer different people in train/validation/test where the dataset permits.
- Keep the face framed consistently across the dataset and webcam input. The starter code resizes the full frame; it does not detect/crop the face automatically.

Use a dataset only if its license and terms permit your intended academic use. Record the dataset name, source, version, and access date in the report. Do not mix frames from unrelated images and call them a genuine video sequence; temporal learning requires chronologically ordered frames from a sequence.

## 3. Train the model

After creating the folder structure and placing your images:

```bash
python train_model.py --data data --epochs 15
```

The best model is saved as `drowsiness_cnn_lstm.keras`. Training prints validation metrics and, if test data is present, a classification report and confusion matrix.

For a quick smoke test, you may lower the number of epochs, but do not report smoke-test results as final research results.

## 4. Run webcam detection

```bash
python webcam_app.py --model drowsiness_cnn_lstm.keras
```

Press **Q** to quit. If the webcam is not camera 0, try:

```bash
python webcam_app.py --camera 1
```

The alert threshold and number of consecutive positive predictions can be adjusted:

```bash
python webcam_app.py --threshold 0.70 --consecutive 3
```

Tune these values only using validation data, then evaluate once on the held-out test set. The displayed score is a model output, not a medically or safety validated probability.

## 5. What to include in the project report

1. Abstract and problem statement
2. Literature survey and comparison of existing methods
3. System requirements
4. Proposed method and system architecture
5. Dataset description, preprocessing, and sequence formation
6. CNN-LSTM architecture and training settings
7. Implementation screenshots
8. Test-set metrics: accuracy, precision, recall, F1-score, and confusion matrix
9. Limitations, conclusion, and future work
10. References and appendix

Do not invent results. Report the actual metrics produced by your held-out test set. Discuss false alarms, missed drowsiness, lighting, glasses, camera angle, and differences between people.

## 6. Three-member task split

- **Member 1 — Data and research:** dataset sourcing, permissions, frame extraction, labeling, literature survey.
- **Member 2 — Model:** training, validation, evaluation, experiments, model architecture.
- **Member 3 — Application and documentation:** webcam app, warning interface, integration, screenshots, report and presentation.

All members should review the complete system and understand the results.

## 7. Known limitations and next improvements

- Face/eye localization is not implemented in this starter version.
- The whole webcam frame is resized, so training and inference framing must match.
- Performance depends on dataset quality and subject diversity.
- This prototype is not validated for real-world driving.
- Future improvements: face/eye crop alignment, blink-duration features, stronger pretrained CNN, subject-independent evaluation, low-light testing, and model optimization.
