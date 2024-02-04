import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = ImageOps.fit(img, (img_height, img_width), Image.ANTIALIAS)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, img_height, img_width, 3))
        prediction = model.predict(img_array)
        label.config(text=str(prediction))

# Load your model
model_path = 'custom_cnn_model_simplified.keras'  # Update with your model's filename
model = load_model(model_path)

# Image dimensions (should be the same as used during training)
img_height, img_width = 224, 224

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to select an image
button = tk.Button(root, text="Select an Image", command=select_image)
button.pack()

# Label to show the classification results
label = tk.Label(root, text="Classification results appear here")
label.pack()

# Run the application
root.mainloop()
