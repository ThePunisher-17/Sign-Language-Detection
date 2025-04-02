# Import necessary libraries
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import tempfile

# Load the trained model
model = models.vgg16(pretrained=False)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 25)
model.load_state_dict(torch.load('sign_language_vgg16_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the labels
label_dir = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24: 'Z'}

# Prediction function
def predict_sign(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
        return None

    st.info("Press 'Space' to capture, 'Esc' to exit")
    captured_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture Image", gray_frame)

        key = cv2.waitKey(1)
        if key == 32:
            captured_image = gray_frame
            break
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, captured_image)
            return temp_file.name
    return None

# Streamlit app
st.title("Sign Language MNIST Classifier")

# Capture Image from Camera
if st.button("Capture Image from Camera"):
    image_path = capture_image()
    if image_path:
        image = Image.open(image_path).convert('L')
        st.image(image, caption='Captured Image', use_container_width=True)
        prediction = predict_sign(image)
        st.write(f"Predicted sign: {label_dir[prediction]}")
    else:
        st.write("No image captured!")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    prediction = predict_sign(image)
    st.write(f"Predicted sign: {label_dir[prediction]}")