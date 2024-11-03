import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import ssl
import time

# Bypass SSL verification for TensorFlow Hub model download
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained style transfer model
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess the style image
def load_style_image():
    style_image = cv2.imread("st.jpg")
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    return tf.image.resize(style_image, (256, 256))

# Load the style image
style_image = load_style_image()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Adjust frame rate by timing
fps = 0
start_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.

    # Resize the input image for faster processing
    img = tf.image.resize(img, (384, 384))

    # Perform style transfer
    outputs = model(tf.constant(img), tf.constant(style_image))
    stylized_image = outputs[0]

    # Post-process the stylized image
    stylized_image = np.squeeze(stylized_image.numpy())
    stylized_image = np.clip(stylized_image * 255, 0, 255).astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)

    # Resize stylized image to match original frame size
    stylized_image = cv2.resize(stylized_image, (frame.shape[1], frame.shape[0]))

    # Display only the stylized frame
    cv2.imshow('Stylized Webcam Feed', stylized_image)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()