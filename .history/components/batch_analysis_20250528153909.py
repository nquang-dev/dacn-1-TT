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
                
                images.append(image)
                cam_images.append(cam_image)
                predictions.append(prediction)
                probs_normal.append(prob_normal)
                probs_tb.append(prob_tb)
                process_times.append(process_time)
                filenames.append(uploaded_file.name)
                pdf_buffers.append(pdf_buffer)
                
                results.append({
                    'Tên file': uploaded_file.name,
                    'Kết quả': 'Lao phổi' if prediction == 1 else 'Bình thường',
                    'Xác suất bình thường': f'{prob_normal:.2%}',
                    'Xác suất lao phổi': f'{prob_tb:.2%}',
                    'Thời gian xử lý': f'{process_time:.2f}s'
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.markdown("""
            <div class="processing-complete">
                <span class="complete-icon">✨</span>
                <span>Phân tích hoàn tất!</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.analysis_results = {
                "results": results, "images": images, "cam_images": cam_images,
                "predictions": predictions, "probs_normal": probs_normal,
                "probs_tb": probs_tb, "process_times": process_times,
                "filenames": filenames, "pdf_buffers": pdf_buffers
            }
        
        # Hiển thị kết quả
        if st.session_state.analysis_results["results"]:
            results = st.session_state.analysis_results["results"]
            predictions = st.session_state.analysis_results["predictions"]
            
            # Thống kê tổng quan
            total = len(predictions)
            normal_count = predictions.count(0)
            tb_count = predictions.count(1)
            
            st.markdown(f"""
            <div class="summary-dashboard">
                <div class="dashboard-card total">
                    <div class="card-icon">📊</div>
                    <div class="card-content">
                        <div class="card-number">{total}</div>
                        <div class="card-label">Tổng số ảnh</div>
                    </div>
                </div>
                <div class="dashboard-card normal">
                    <div class="card-icon">✅</div>
                    <div class="card-content">
                        <div class="card-number">{normal_count}</div>
                        <div class="card-label">Bình thường</div>
                    </div>
                </div>
                <div class="dashboard-card warning">
                    <div class="card-icon">⚠️</div>
                    <div class="card-content">
                        <div class="card-number">{tb_count}</div>
                        <div class="card-label">Cần chú ý</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bảng kết quả
            df = pd.DataFrame(results)
            st.markdown('<div class="results-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Hiển thị các trường hợp cần chú ý
            if tb_count > 0:
                st.markdown("""
                <div class="attention-section">
                    <h3>🔍 Các trường hợp cần chú ý</h3>
                </div>
                """, unsafe_allow_html=True)
                
                images = st.session_state.analysis_results["images"]
                cam_images = st.session_state.analysis_results["cam_images"]
                filenames = st.session_state.analysis_results["filenames"]
                
                for idx, (img, cam_img, pred, filename) in enumerate(zip(images, cam_images, predictions, filenames)):
                    if pred == 1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img, caption=f"📸 {filename}", use_container_width=True)
                        with col2:
                            st.image(cam_img, caption=f"🎯 CAM - {filename}", use_container_width=True)
            
            # PDF Downloads
            st.markdown("""
            <div class="download-section">
                <h3>📋 Tải báo cáo PDF</h3>
            </div>
            """, unsafe_allow_html=True)
            
            pdf_buffers = st.session_state.analysis_results["pdf_buffers"]
            filenames = st.session_state.analysis_results["filenames"]
            
            for idx, (filename, pdf_buffer) in enumerate(zip(filenames, pdf_buffers)):
                pdf_col1, pdf_col2 = st.columns([3, 1])
                with pdf_col1:
                    result_text = "Cần chú ý" if predictions[idx] == 1 else "Bình thường"
                    status_class = "warning" if predictions[idx] == 1 else "normal"
                    st.markdown(f"""
                    <div class="pdf-download-item">
                        <div class="file-info">
                            <span class="file-name">📄 {filename}</span>
                            <span class="file-status {status_class}">{result_text}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with pdf_col2:
                    st.download_button(
                        label="📋 Tải PDF",
                        data=pdf_buffer,
                        file_name=f"bao_cao_{filename.split('.')[0]}.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{idx}"
                    )

