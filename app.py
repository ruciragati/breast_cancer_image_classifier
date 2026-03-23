import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import urllib.request
import os

MODEL_URL = "https://huggingface.co/datasets/guccirucci/breast-cancer-image-classifier/resolve/main/model.tflite"
MODEL_PATH = "model.tflite"

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🎗️")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Please wait..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🎗️ Breast Cancer Image Classifier")
st.write("Upload a medical scan to analyze for Malignant vs. Benign characteristics.")

uploaded_file = st.file_uploader("Upload Scan (JPG/PNG/JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB').resize((256, 256))
    st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing..."):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            label = "Malignant" if prediction > 0.5 else "Benign"
            confidence = prediction if prediction > 0.5 else (1 - prediction)
            
            st.subheader(f"Result: {label}")
            st.info(f"Confidence Level: {confidence:.2%}")

st.divider()
st.caption("**Disclaimer:** This AI tool is for educational purposes only. "
           "It is not a substitute for professional medical diagnosis or consultation.")