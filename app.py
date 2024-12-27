import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
from PIL import Image

# Load the pre-trained model from the repository files
@st.cache_resource
def load_model():
    # Ensure the model files are in the same directory as this script
    try:
        # Load model architecture
        with open("emotiondetector.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        # Load model weights
        model.load_weights("emotiondetector.h5")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load Haar Cascade for face detection
@st.cache_resource
def load_haar_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

# Preprocess the image for prediction
def preprocess_image(image, face_coords):
    x, y, w, h = face_coords
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))  # Resize to 48x48 pixels
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face = face.reshape(1, 48, 48, 1) / 255.0  # Normalize and reshape
    return face

# Main Streamlit app
def main():
    st.title("Facial Emotion Detection")
    st.write("Upload an image, and the app will detect facial emotions on detected faces!")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to OpenCV format
        image = np.array(image)
        if image.ndim == 2 or image.shape[2] == 1:  # Handle grayscale images
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # Handle RGBA images
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Load model and cascade
        model = load_model()
        face_cascade = load_haar_cascade()
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        if model is None or face_cascade.empty():
            st.error("Error: Model or Haar Cascade could not be loaded.")
            return

        # Detect faces
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            st.write("No faces detected.")
        else:
            for (x, y, w, h) in faces:
                # Preprocess each detected face
                try:
                    face_input = preprocess_image(image, (x, y, w, h))
                    prediction = model.predict(face_input)
                    emotion_label = labels[np.argmax(prediction)]

                    # Draw rectangle around the face and add emotion label
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                except Exception as e:
                    st.error(f"Error processing face at ({x}, {y}): {e}")

            # Convert image back to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Processed Image with Detected Emotions", use_column_width=True)

if __name__ == "__main__":
    main()
