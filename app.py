import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch
import cv2
import detect

st.title("YUMMY")

st.write("Upload your Image...")

# model = torch.hub.load('./yolov5', 'custom', path='./last.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/Model2.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    image = cv2.imdecode(file_bytes, 1)

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # st.image(imgRGB)

    st.write("")
    st.write("Detecting...")
    result = model(imgRGB, size=600)

    detect_class = result.pandas().xyxy[0]

    detect_class['name'] = detect_class['name'].map({'Darathong': 'Darathong',
                                                     'SaneCharn': 'SaneCharn',
                                                     'ChorMuang': 'ChorMuang'})

    # Get unique names
    unique_names = detect_class['name'].unique()

    # Display the unique names without numbers
    st.write(unique_names)
    # for name in unique_names:
    #     if name == 'Darathong':
    #         st.text('TH: ดาราทอง  EN: Darathong')
    #     elif name == 'SaneCharn':
    #         st.text('TH: เสน่ห์จันทร์  EN: SaneCharn')
    #     elif name == 'ChorMuang':
    #         st.text('TH: ช่อม่วง  EN: ChorMuang')
            
    #     image_path = f"data/images/{name}.jpg"
    #     if os.path.exists(image_path):
    #         st.image(Image.open(image_path), caption='Original Image', use_column_width=True)
    
    # if unique_names == ['ดาราทอง (Darathong)']:
    #     st.image(Image.open("data/images/Darathong.jpg"), caption='Original Image', use_column_width=True)
    # if unique_names == ['เสน่ห์จันทร์ (SaneCharn)']:
    #     st.image(Image.open("data/images/SaneCharn.jpg"), caption='Original Image', use_column_width=True)
    # if unique_names == ['ช่อม่วง (ChorMuang)']:
    #     st.image(Image.open("data/images/ChorMuang.jpg"), caption='Original Image', use_column_width=True)
    
