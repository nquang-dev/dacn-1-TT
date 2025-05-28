import streamlit as st
from PIL import Image
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import apply_cam
from .shared_utils import create_pdf_report, register_fonts

def render_batch_analysis(model, last_conv_layer, preprocess):
    """Render tab phân tích hàng loạt"""
    
    # Register fonts
    register_fonts()
    
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">📊</span>
            <h2>Phân tích hàng loạt</h2>
        </div>
        <p class="section-desc">Tải lên nhiều ảnh X-quang để phân tích đồng thời và so sánh kết quả</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "uploaded_files_multiple" not in st.session_state:
        st.session_state.uploaded_files_multiple = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {
            "results": [], "images": [], "cam_images": [], "predictions": [],
            "probs_normal": [], "probs_tb": [], "process_times": [], 
            "filenames": [], "pdf_buffers": []
        }

    # Upload zone
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3>📁 Tải lên nhiều ảnh X-quang</h3>
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
    
    if uploaded_files:
        st.session_state.uploaded_files_multiple = uploaded_files
        
        st.markdown(f"""
        <div class="upload-summary">
            <span class="summary-icon">📁</span>
            <span class="summary-text">Đã tải lên {len(uploaded_files)} ảnh</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Preview grid
        st.markdown('<div class="preview-grid">', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[:8]):
            with cols[i % 4]:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        if len(uploaded_files) > 8:
            st.markdown(f'<div class="more-files">+{len(uploaded_files) - 8} ảnh khác</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            analyze_all = st.button("🚀 Phân tích tất cả", key="analyze_multiple", type="primary")
        with action_col2:
            reset_all = st.button("🔄 Làm mới", key="reset_multiple")
        
        if reset_all:
            for key in ["multiple_upload", "uploaded_files_multiple", "analysis_results"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if analyze_all:
            results = []
            images = []
            cam_images = []
            predictions = []
            probs_normal = []
            probs_tb = []
            process_times = []
            filenames = []
            pdf_buffers = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(f"""
                <div class="processing-status">
                    <span class="status-icon">🔬</span>
                    <span>Đang xử lý {i+1}/{len(uploaded_files)}: {uploaded_file.name}</span>
                </div>
                """, unsafe_allow_html=True)
                
                image = Image.open(uploaded_file).convert('RGB')
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                image.save(temp_file_path)
                
                cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                    temp_file_path, model, preprocess, last_conv_layer)
                
                pdf_buffer = create_pdf_report(
                    image, cam_image, prediction, prob_normal, prob_tb, process_time, uploaded_file.name)
                
                images.appen
