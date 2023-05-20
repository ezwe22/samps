import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import string

# Load the Alphabet Classification model
model = tf.keras.models.load_model('alphabet_model.h5')

class_names = list(string.ascii_uppercase)

def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match input size
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = image / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@tf.function
def import_and_predict(image_data, model):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    return class_label

st.write("# Alphabet Classifier/Recognizer")
file = st.file_uploader("Select an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    class_label = import_and_predict(image, model)
    st.success("Classification: " + class_label)
