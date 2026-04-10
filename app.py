import os
os.environ["OPENCV_HEADLESS"] = "1"
os.environ["DISPLAY"] = ":0"
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Advanced YOLO Robotic System", layout="wide")
st.title("Advanced YOLOv8 Intelligent Robotic Pick-and-Place System")
st.title("DONE BY JAY KUMAR")
st.caption("**BTech CSE Final Year Project** | Enhanced Version")

# Sidebar Controls
st.sidebar.header("⚙️ Model Settings")
model_size = st.sidebar.selectbox("Model Size", ["yolov8n.pt", "yolov8s.pt"], index=0)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_size)

st.subheader("Live Object Detection + Robotic Arm Simulation")

option = st.radio("Choose Input", ["Upload Image", "Use Webcam"], horizontal=True)

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        image = Image.open(uploaded)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        
        with col2:
            start_time = time.time()
            
            results = model(img_array, conf=conf_threshold)[0]
            annotated = results.plot()
            
            inference_time = (time.time() - start_time) * 1000
            fps = 1000 / inference_time if inference_time > 0 else 0
            
            st.image(annotated, caption="✅ YOLOv8 Detection + Robotic Simulation")
            
            st.subheader("📊 Detection Metrics")
            st.write(f"**Inference Time:** {inference_time:.2f} ms | **FPS:** {fps:.1f}")
            
            st.subheader("🤖 Simulated Robotic Arm Actions")
            detected_objects = 0
            for box in results.boxes:
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)
                st.success(f"**{cls_name}** (Confidence: {conf:.2f}) → Robotic Arm will **PICK** & **PLACE**")
                detected_objects += 1
            
            if detected_objects == 0:
                st.warning("No objects detected above confidence threshold.")
            
            # Download
            annotated_pil = Image.fromarray(annotated)
            st.download_button("📥 Download Annotated Image", 
                             data=annotated_pil.tobytes(), 
                             file_name="advanced_yolo_detection.png", 
                             mime="image/png")

elif option == "Use Webcam":
    st.warning("Webcam not supported in WSL. Use Upload Image.")

st.divider()
st.caption("**Resume Highlights:** Real-time Object Detection | Robotic Simulation | Confidence Control | Performance Metrics | Customizable Model")
