import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    with open("emotiondetector.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("emotiondetector.h5")
    return model

# Load Haar cascade
@st.cache_resource
def load_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

model = load_model()
face_cascade = load_cascade()

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

st.title("Face Emotion Detection App")
st.write("Upload a grayscale image to detect emotions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert('L')
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect faces
        faces = face_cascade.detectMultiScale(image_np, 1.3, 5)

        if len(faces) == 0:
            st.error("No face detected in the image.")
        else:
            for (p, q, r, s) in faces:
                face = image_np[q:q + s, p:p + r]
                face_resized = cv2.resize(face, (48, 48))
                img = extract_features(face_resized)

                # Make prediction
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]

                # Display results
                st.success(f"Detected Emotion: {prediction_label}")
                st.image(face_resized, caption="Detected Face", use_column_width=False)
    except Exception as e:
        st.error(f"An error occurred: {e}")
