import os
# Tắt thống báo từ TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TERM"] = "dumb"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Tạo DataFrame chứa đường dẫn ảnh
def load_img_paths(base_dir):
    data = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir) and label.strip():
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(label_dir, fname)
                    data.append({'filepath': fpath, 'label': label})
    return pd.DataFrame(data)

# Thống kê số lượng ảnh của mỗi lớp và trả về dict 
def label_stats(dir_path):
    label_counts = {}
    for dirpath, _, filenames in os.walk(dir_path):
        if dirpath == dir_path:
            continue
        label = os.path.basename(dirpath)
        count = len([f for f in filenames if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"Label: '{label}' - {count} images")
        label_counts[label] = count
    return label_counts

# Tạo bộ nạp dữ liệu cho Train, Validation, Test từ DataFrame đầu vào
def create_data_generators(train_df, val_df, test_df, target_size=(224, 224), batch_size=32):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_loader = train_gen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=target_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )
    val_loader = val_gen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=target_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    test_loader = test_gen.flow_from_dataframe(
        test_df, x_col='filepath', y_col='label',
        target_size=target_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return train_loader, val_loader, test_loader

# Xây dựng mô hình ResNet50
def build_resnet_model(num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình
def train_model(model, train_loader, val_loader, model_path='resnet50.h5', epochs=50):
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    return history


# Dự đoán nhãn trên tập test
def evaluate_model(model, test_loader):
    pred_prob = model.predict(test_loader)
    pred_labels = np.argmax(pred_prob, axis=1)
    true_labels = test_loader.classes
    class_names = list(test_loader.class_indices.keys())

    report = classification_report(true_labels, pred_labels, target_names=class_names)
    cm = confusion_matrix(true_labels, pred_labels)
    return report, cm, class_names

# Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Phần thực thi
if __name__ == "__main__":
    # Cấu hình cơ bản
    BATCH_SIZE = 32
    TARGET_SIZE = (224, 224)
    train_dir = 'E:/DATN/Dataset_kaggle/RiceLeafsDisease/train'
    val_dir = 'E:/DATN/Dataset_kaggle/RiceLeafsDisease/validation'

    #Thống kê số lượng ảnh của mỗi nhãn
    print("LABEL STATISTICS")
    print("\nLABEL TRAIN")
    label_stats(train_dir)
    print("\nLABEL VALIDATION")
    label_stats(val_dir)

    # Tải đường dẫn ảnh và gộp thành 1 DataFrame
    print("\nLOADING IMAGE PATHS")
    train_data = load_img_paths(train_dir)
    val_data = load_img_paths(val_dir)
    df = pd.concat([train_data, val_data], ignore_index=True)

    # Chia dữ liệu
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Tạo dữ liệu
    print("\nCREATING DATA LOADERS")
    train_loader, val_loader, test_loader = create_data_generators(train_df, val_df, test_df,
                                                                    target_size=TARGET_SIZE,
                                                                    batch_size=BATCH_SIZE)
    # Xác định số lớp (nhãn)
    num_classes = len(train_loader.class_indices)

    # Xây dựng và huấn luyện mô hình
    print("\nTRAINING MODEL")
    model = build_resnet_model(num_classes)
    history = train_model(model, train_loader, val_loader)

    # Đánh giá mô hình
    print("\nEVALUATING MODEL")
    report, cm, class_names = evaluate_model(model, test_loader)
    print(report)
    plot_confusion_matrix(cm, class_names)

    # Vẽ biểu đồ accuracy và loss
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
