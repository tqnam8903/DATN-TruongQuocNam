import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load mô hình ResNet và CNN
resnet_model = load_model("resnet50.h5")
cnn_model = load_model("cnn.h5")

# Danh sách nhãn
class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']

current_image_path = None  # Biến lưu ảnh hiện tại

def browser_img():
    global current_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        current_image_path = file_path
        img = Image.open(file_path).resize((224, 224))
        img_tk = ImageTk.PhotoImage(img)
        input_image_label.config(image=img_tk)
        input_image_label.image = img_tk

def detect_disease():
    if not current_image_path:
        disease_result.set("Vui lòng chọn ảnh.")
        return
    
    selected_model = model_choice.get()
    if not selected_model:
        disease_result.set("Vui lòng chọn mô hình.")
        return

    try:
        img = Image.open(current_image_path).resize((224, 224))

        # ResNet Tiền xử lý
        img_resnet = img_to_array(img)
        img_resnet = preprocess_input(img_resnet)
        img_resnet = np.expand_dims(img_resnet, axis=0)

        # CNN Tiền xử lý
        img_cnn = img_to_array(img) / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)

        result_text = ""

        if selected_model == "ResNet50 only":
            resnet_pred = resnet_model.predict(img_resnet)
            resnet_label = np.argmax(resnet_pred)
            result_text = f"ResNet: {class_labels[resnet_label]}"

        elif selected_model == "CNN only":
            cnn_pred = cnn_model.predict(img_cnn)
            cnn_label = np.argmax(cnn_pred)
            result_text = f"CNN: {class_labels[cnn_label]}"

        disease_result.set(result_text)
    except Exception as e:
        traceback.print_exc()
        disease_result.set("Lỗi: Không thể dự đoán.")

def reset():
    global current_image_path
    current_image_path = None
    input_image_label.config(image="")
    input_image_label.image = None
    disease_result.set("")
    model_choice.set("")  

def exit():
    root.quit()

# UI setup
root = tk.Tk()
root.title("Rice Leaf Disease Classification")
root.configure(bg="white")
root.geometry("1000x700")
root.minsize(800, 600)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (1000 / 2))
y = int((screen_height / 2) - (700 / 2))
root.geometry(f"1000x700+{x}+{y}")

root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=7)
root.rowconfigure(2, weight=2)
root.columnconfigure(0, weight=1)

# Header
header_frame = tk.Frame(root, bg="white", bd=3, relief=tk.GROOVE)
header_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
header_frame.columnconfigure(0, weight=1)
header_frame.columnconfigure(1, weight=4)

logo_frame = tk.Frame(header_frame, bg="white")
logo_frame.grid(row=0, column=0, sticky="w", padx=10)

logo_img_raw = Image.open("Image/logo.jpg").resize((250, 150))
logo_img = ImageTk.PhotoImage(logo_img_raw)
logo_label = tk.Label(logo_frame, image=logo_img, bg="white")
logo_label.image = logo_img
logo_label.pack()

info_frame = tk.Frame(header_frame, bg="white")
info_frame.grid(row=0, column=1, sticky="e", padx=10)
tk.Label(info_frame, text='RICE LEAF DISEASE CLASSIFICATION', font=("Arial", 20, "bold"), bg="white").pack(anchor="center", pady=(0, 8))
tk.Label(info_frame, text="Họ và tên: Trương Quốc Nam", font=("Arial", 12, "bold"), bg="white").pack(anchor="e")
tk.Label(info_frame, text="Lớp: 63CNTT2", font=("Arial", 12, "bold"), bg="white").pack(anchor="e")
tk.Label(info_frame, text="MSV: 2151060299", font=("Arial", 12, "bold"), bg="white").pack(anchor="e")

# Middle
middle_frame = tk.Frame(root, bg="white")
middle_frame.grid(row=1, column=0, sticky="nsew", pady=5)
middle_frame.columnconfigure(0, weight=8)
middle_frame.columnconfigure(1, weight=2)
middle_frame.rowconfigure(0, weight=1)

# Image
image_frame = tk.Frame(middle_frame, bg="#FFFFE0", bd=3, relief=tk.GROOVE)
image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
image_container = tk.Frame(image_frame, width=240, height=360, bg="white")
image_container.pack(pady=10)
image_container.pack_propagate(False)
input_image_label = tk.Label(image_container, bg="white")
input_image_label.pack(expand=True)
image_caption = tk.Label(image_frame, text="Browser Input", font=("Arial", 12, "bold"), bg="#FFFFE0")
image_caption.pack(pady=(5, 5))

# Buttons + Combobox
button_frame = tk.Frame(middle_frame, bg="#FFFFE0", bd=3, relief=tk.GROOVE)
button_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
button_frame.rowconfigure([0, 1, 2, 3, 4, 5], weight=1)
button_frame.columnconfigure(0, weight=1)

btn_style = {"font": ("Arial", 12, "bold"), "width": 15, "bg": "#FF6666"}
tk.Button(button_frame, text="Browser Input", command=browser_img, **btn_style).grid(row=0, column=0, pady=10, padx=20, sticky="ew")
tk.Button(button_frame, text="Detection", command=detect_disease, **btn_style).grid(row=1, column=0, pady=10, padx=20, sticky="ew")
tk.Button(button_frame, text="Reset", command=reset, **btn_style).grid(row=2, column=0, pady=10, padx=20, sticky="ew")
tk.Button(button_frame, text="Exit", command=exit, **btn_style).grid(row=3, column=0, pady=10, padx=20, sticky="ew")

# Combobox để chọn mô hình 
tk.Label(button_frame, text="Select Model", font=("Arial", 12, "bold"), bg="#FFFFE0").grid(row=4, column=0, pady=(10, 5))
model_choice = tk.StringVar(value="")  # mặc định là rỗng
model_combobox = ttk.Combobox(button_frame, textvariable=model_choice, font=("Arial", 11), state="readonly", width=17)
model_combobox['values'] = ["ResNet50 Model", "CNN Model"]
model_combobox.grid(row=5, column=0, padx=20, pady=(0, 10))

# Bottom
bottom_frame = tk.Frame(root, bg="#FFFFE0", bd=3, relief=tk.GROOVE)
bottom_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
tk.Label(bottom_frame, text="Classification", font=("Arial", 14, "bold"), bg="#FFFFE0").pack(pady=(10, 5))
disease_result = tk.StringVar()
disease_entry = tk.Entry(bottom_frame, textvariable=disease_result, font=("Arial", 12), fg="green", bg="white", width=40, justify='center', relief=tk.SUNKEN)
disease_entry.pack(pady=(0, 10))

root.mainloop()
