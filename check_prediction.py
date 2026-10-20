
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = "models/best_eye_model.keras"
TEST_PATH = "data/eye_state_split/test"

IMG_SIZE = (93, 93)
BATCH_SIZE = 32

print("Loading model...")
model = load_model(MODEL_PATH)

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)

dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = dataset.class_names
print("Class order:", class_names)

all_actual = []
all_predicted = []
all_probabilities = []

for images, labels in dataset:
    # Model-la Rescaling(1/255) layer irundhaal
    # manual-ah /255 panna vendam.
    probabilities = model(images, training=False).numpy().reshape(-1)

    actual = labels.numpy().reshape(-1).astype(int)
    predicted = (probabilities >= 0.5).astype(int)

    all_actual.extend(actual)
    all_predicted.extend(predicted)
    all_probabilities.extend(probabilities)

all_actual = np.array(all_actual)
all_predicted = np.array(all_predicted)

accuracy = np.mean(all_actual == all_predicted)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(
    all_actual,
    all_predicted,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(all_actual, all_predicted, labels=[0, 1]))

print("\nSample predictions:")
for i in range(min(10, len(all_actual))):
    print(
        f"Actual: {class_names[all_actual[i]]} | "
        f"Open probability: {all_probabilities[i]:.4f} | "
        f"Predicted: {class_names[all_predicted[i]]}"
    )
