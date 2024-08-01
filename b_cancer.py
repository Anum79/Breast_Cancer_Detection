import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    # Convert the PIL image to a numpy array
    image = np.array(image)
    # Resize the image to the input size of the model using OpenCV
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
    # Normalize the image data to [0, 1]
    image = image / 255.0
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.title("Cancer Prediction App")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_image is not None:
        # Open the image file
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)

        # Map the predicted class to the corresponding label
        class_labels = ['Benign', 'Malignant', 'Normal']
        predicted_label = class_labels[predicted_class]

        # Display the result
        st.write(f'Predicted class: {predicted_label}')

if __name__ == "__main__":
    main()
