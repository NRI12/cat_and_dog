import numpy as np
import cv2
import keras
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

CATEGORIES = ['Cat', 'Dog']

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr

def predict_image():
    file_path = filedialog.askopenfilename()  
    if file_path:
        img_array = image(file_path)
        prediction = model.predict([img_array])
        result_label.config(text=f"Prediction: {CATEGORIES[prediction.argmax()]}")

        img = cv2.imread(file_path)
        img = cv2.resize(img, (300, 300))
        cv2.rectangle(img, (0, 0), (300, 40), (0, 0, 255), -1)
        cv2.putText(img, f"Prediction: {CATEGORIES[prediction.argmax()]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        result_image_label.config(image=img)
        result_image_label.image = img

window = tk.Tk()
window.title("Cat and Dog Classifier")

window.geometry("800x600+100+100")

upload_button = tk.Button(window, text="Upload Image", command=predict_image)
upload_button.pack(pady=20)

result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack()

result_image_label = tk.Label(window)
result_image_label.pack()

#Lưu ý đường dẫn model đã train or train lại
model = keras.models.load_model('model_trained')

window.mainloop()
