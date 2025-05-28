import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from utils.pdf_generator import create_pdf_report
from src.visualization import apply_cam

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def show_batch_analysis(model, last_conv_layer):
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">\U0001F4CA</span>
            <h2>Phân tích hàng loạt</h2>
        </div>
        <p class="section-desc">Tải lên nhiều ảnh X-quang để phân tích đồng thời và so sánh kết quả</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Khởi tạo session state
    if "uploaded_files_multiple" not in st.session_state:
        st.session_state.uploaded_files_multiple = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {
            "results": [], "images": [], "cam_images": [], "predictions": [],
            "probs_normal": [], "probs_tb": [], "process_times": [], 
            "filenames": [], "pdf_buffers": []
        }

    # Upload zone mới
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3>\U0001F4C1 Tải lên nhiều ảnh X-quang</h3>
            <p>Chọn nhiều file cùng lúc để phân tích hàng loạt</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Chọn nhiều ảnh X-quang",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multiple_upload"
    )
    
    # Phần còn lại của mã cho phân tích hàng loạt...
    # (Mã còn lại được giữ nguyên từ app.py gốc)
    
    # Ví dụ:
    if uploaded_files:
        st.session_state.uploaded_files_multiple = uploaded_files
        
        st.markdown(f"""
        <div class="upload-summary">
            <span class="summary-icon">\U0001F4C1</span>
            <span class="summary-text">Đã tải lên {len(uploaded_files)} ảnh</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Các phần còn lại của mã...
