import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TERM"] = "dumb"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    Flatten, Dense, Dropout, Add, ZeroPadding2D,
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_img_paths(base_dir):
    data = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(label_dir, fname)
                    data.append({'filepath': fpath, 'label': label})
    return data

def label_stats(dir_path):
    label_counts = {}
    for dirpath, dirnames, filenames in os.walk(dir_path):
        if dirpath == dir_path:
            continue
        label = os.path.basename(dirpath)
        count = len([f for f in filenames if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Label: '{label}' - {count} images")
        label_counts[label] = count
    return label_counts

BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
train_dir = 'E:/DATN/Dataset_kaggle/RiceLeafsDisease/train'
val_dir = 'E:/DATN/Dataset_kaggle/RiceLeafsDisease/validation'
label_stats(train_dir)
label_stats(val_dir)
train_data = load_img_paths(train_dir)
val_data = load_img_paths(val_dir)

df = pd.DataFrame(train_data + val_data)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    channel_shift_range=10,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

total_classes = len(train_df['label'].unique())

train_img = train_gen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label', target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_img = val_gen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label', target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_img = test_gen.flow_from_dataframe(
    test_df, x_col='filepath', y_col='label', target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(total_classes, activation='softmax')
])

model.compile(optimizer = Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

file_path = "resnet50.h5"
checkpoint = ModelCheckpoint(file_path, monitor = "val_loss", save_best_only = True, save_weights_only = False, verbose = 1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    train_img,
    validation_data=val_img,
    epochs=50,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

plt.plot(history.history['accuracy'], label='Training accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation accuracy', color='red')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation loss', color='red')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

pred_prob = model.predict(test_img)
pred_labels = np.argmax(pred_prob, axis=1)
true_labels = test_img.classes
class_names = list(test_img.class_indices.keys())

report = classification_report(true_labels, pred_labels, target_names=class_names)
print("Classification Report:\n")
print(report)

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
