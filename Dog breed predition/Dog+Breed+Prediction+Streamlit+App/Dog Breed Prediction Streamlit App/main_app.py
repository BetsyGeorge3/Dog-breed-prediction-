# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading the Model
model = load_model('dog_breed.h5')

# Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Set Page Configuration
st.set_page_config(
    page_title="Dog Breed Prediction",
    page_icon="🐾",
    layout="centered",
)

# Custom CSS Style
custom_style = """
    <style>
        body {
            background-color: #f0f0f0;
            color: #333;
        }
        .st-bd {
            max-width: 800px;
        }
        .st-ef {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')

# Function to make prediction
def make_prediction(model, image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Make Prediction
    predictions = model.predict(image)
    return predictions

# On predict button click
if submit:
    if dog_image is not None:
        try:
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Display the image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")

            # Make Prediction
            with st.spinner("Making Prediction..."):
                predictions = make_prediction(model, opencv_image)

            # Display Prediction
            breed_index = np.argmax(predictions)
            breed_confidence = predictions[0, breed_index]
            st.success(f"The predicted breed is {CLASS_NAMES[breed_index]} with confidence: {breed_confidence:.2%}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

