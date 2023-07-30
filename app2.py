import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model_accuracy98%(2).h5')
class_names = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']

def classify_image(image):
    # Resize the image to the desired input shape
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image1 = image.astype('uint8')
    image2 = np.expand_dims(image1, 0)
    # Make predictions using the model
    predictions = model.predict(image2)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    return class_names[class_index], confidence

def main():
    st.title("Maize Leaf Disease Classification")
    st.text("Upload an image of a maize leaf to classify its disease.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the image from the uploader
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and get predictions
        class_name, confidence = classify_image(image)

        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()
