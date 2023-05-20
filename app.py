import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Traffic Sign Classification model
traffic_model = tf.keras.models.load_model('my_model.h5')

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((32, 32))  # Resize image to match input size
    image = np.array(image)  # Convert PIL Image to NumPy array
    image = image / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def import_and_predict(image_data, model):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    return class_label

model = load_model()
class_names = ['Stop', 'Yield', 'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60',
               'Speed Limit 70', 'Speed Limit 80', 'No Overtaking', 'No Entry', 'Road Work']

st.write("# Traffic Sign Classification")
file = st.file_uploader("Select an image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    class_label = import_and_predict(image, model)
    st.success("Classification: " + class_label)
