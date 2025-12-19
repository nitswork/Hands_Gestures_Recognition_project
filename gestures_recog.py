import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import os
import mediapipe as mp

# ------------------ STEP 1: TRAIN THE MODEL ------------------

data_dir = "data"
model_path = "gesture_mobilenet_model.h5"

if not os.path.exists(model_path):
    print("Training model with MobileNetV2...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    class_names = list(train_data.class_indices.keys())
    print(f"Classes detected: {class_names}")

    # Base model from MobileNetV2
    base_model = MobileNetV2(input_shape=(128, 128, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False  # freeze pretrained layers

    # Add custom layers for gesture recognition
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint("best_mobilenet_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        callbacks=[checkpoint]
    )

    model.save(model_path)
    print("Model trained and saved as gesture_mobilenet_model.h5")

else:
    print("Pretrained model found — skipping training.")

# ------------------ STEP 2: LOAD MODEL FOR PREDICTION ------------------

print("Loading trained model...")
model = tf.keras.models.load_model("best_mobilenet_model.h5" if os.path.exists("best_mobilenet_model.h5") else model_path)
class_names = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
print(f"Classes detected: {class_names}")

# ------------------ STEP 3: REAL-TIME PREDICTION ------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Starting webcam — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = cv2.resize(hand_img, (128, 128))
                img = np.expand_dims(hand_img, axis=0) / 255.0

                prediction = model.predict(img, verbose=0)
                gesture = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                text = f"{gesture} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited successfully.")
