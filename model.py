# model.py
"""
Train an emotion recognition model.
Usage:
 - Prepare training data as directories: data/train/<emotion_label>/*.jpg and data/val/<emotion_label>/*.jpg
 - Or adapt the code to load FER2013 CSV.
 - Run: python model.py
This will save the trained model to models/creative_model_name.h5
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

OUTPUT_MODEL_PATH = "models/creative_model_name.h5"
IMG_SIZE = (48, 48)   # common size for emotion datasets
BATCH_SIZE = 64
EPOCHS = 40

def build_model(input_shape=(48,48,1), n_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    # Expect data/train/<label>/*.jpg and data/val/<label>/*.jpg
    if not os.path.exists("data/train"):
        print("No training data found in data/train. Please prepare dataset or load an existing model.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "data/train",
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        "data/val",
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    n_classes = train_generator.num_classes
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), n_classes=n_classes)
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early]
    )
    print("Training complete. Best model saved to:", OUTPUT_MODEL_PATH)

def quick_test_load(model_path=OUTPUT_MODEL_PATH):
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Loaded model:", model_path)
        print(model.summary())
    else:
        print("No model found at", model_path)

if __name__ == "__main__":
    # Run training if data exists; otherwise show how to test load
    train()
