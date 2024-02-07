import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model_path = 'custom_cnn_model_improved_v3.keras'
model = load_model(model_path)
print(f"Loaded model from {model_path}")

# Directory containing images to be classified
images_dir = r'C:\Users\Aron Jintalan\Desktop\Service Learning Dataset\Test\Sambong'
print(f"Looking for images in {images_dir}")

# Image dimensions
img_height, img_width = 224, 224

# Dictionary to label all plant classes.
labels = {0: 'sambong', 1: 'akapulko', 2: 'lagundi'}

# Function to prepare an image for classification
def prepare_image(file):
    img = image.load_img(file, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.  # Scale image values
    return img_array

# Iterate over images in the directory
for img_file in os.listdir(images_dir):
    img_path = os.path.join(images_dir, img_file)
    if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".jpg")):  # Check for image files
        print(f"Processing image: {img_path}")
        img_prepared = prepare_image(img_path)
        prediction = model.predict(img_prepared)
        predicted_class = labels[np.argmax(prediction)]
        print(f"Image {img_file} is predicted as {predicted_class} with confidence {np.max(prediction)}")
