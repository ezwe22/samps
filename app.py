import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Beauty Classification model
beauty_model = tf.keras.models.load_model('beauty_model.h5')

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('beauty_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match input size
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = image / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def import_and_predict(image_data, model):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    if prediction < 0.5:
        class_label = "Average"
    else:
        class_label = "Beautiful"
    return class_label

model = load_model()

st.write("# Beauty Classification: Beautiful or Average")
file = st.file_uploader("Select an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    class_label = import_and_predict(image, model)
    st.success("Classification: " + class_label)
