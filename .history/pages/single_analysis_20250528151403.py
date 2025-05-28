import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

def single_analysis_page(model, preprocess, last_conv_layer, create_pdf_report):
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">🔍</span>
            <h2>Phân tích ảnh X-quang đơn lẻ</h2>
        </div>
        <p class="section-desc">Tải lên một ảnh X-quang để phân tích và nhận kết quả chẩn đoán chi tiết</p>
    </div>
    """, unsafe_allow_html=True)

    with open('styles/single_analysis.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    upload_col, preview_col = st.columns([1, 1])

    with upload_col:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-header">
                <h3>📁 Tải lên ảnh X-quang</h3>
                <p>Hỗ trợ định dạng: JPG, JPEG, PNG</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Chọn ảnh X-quang",
            type=["jpg", "jpeg", "png"],
            key="single_upload",
            help="Kéo thả ảnh vào đây hoặc nhấn để chọn file"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                analyze_btn = st.button("🚀 Bắt đầu phân tích", key="analyze_single", type="primary")
            with btn_col2:
                reset_btn = st.button("🔄 Làm mới", key="reset_single")

            if reset_btn:
                st.rerun()

            if analyze_btn:
                with st.spinner("🔬 Đang phân tích..."):
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    image.save(temp_file_path)

                    from src.visualization import apply_cam
                    cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                        temp_file_path, model, preprocess, last_conv_layer)

                    with preview_col:
                        if prediction == 1:
                            st.markdown("""
                            <div class="result-alert danger">
                                <div class="alert-icon">⚠️</div>
                                <div class="alert-content">
                                    <h3>Phát hiện bất thường</h3>
                                    <p>Có dấu hiệu nghi ngờ lao phổi</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="result-alert success">
                                <div class="alert-icon">✅</div>
                                <div class="alert-content">
                                    <h3>Kết quả bình thường</h3>
                                    <p>Không phát hiện dấu hiệu bất thường</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown('<div class="image-viewer">', unsafe_allow_html=True)
                        st.image(cam_image, caption="🎯 Vùng phân tích CAM", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(8, 4))
                        categories = ['Bình thường', 'Lao phổi']
                        values = [prob_normal, prob_tb]
                        colors = ['#00D4AA', '#FF6B6B']
                        bars = ax.barh(categories, values, color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Xác suất')
                        ax.set_title('📊 Phân tích xác suất', fontweight='bold')
                        for i, (bar, value) in enumerate(zip(bars, values)):
                            ax.text(value + 0.02, i, f'{value:.1%}', va='center', fontweight='bold')
                        st.pyplot(fig)

                        st.markdown(f"""
                        <div class="info-panel">
                            <div class="info-row">
                                <span class="info-label">⏱️ Thời gian xử lý:</span>
                                <span class="info-value">{process_time:.2f}s</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">📄 Tên file:</span>
                                <span class="info-value">{uploaded_file.name}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        pdf_buffer = create_pdf_report(
                            image, cam_image, prediction, prob_normal, prob_tb, process_time, uploaded_file.name)

                        st.download_button(
                            label="📋 Tải báo cáo PDF",
                            data=pdf_buffer,
                            file_name=f"bao_cao_{uploaded_file.name.split('.')[0]}.pdf",
                            mime="application/pdf",
                            key="download_single_pdf"
                        )

    with preview_col:
        if uploaded_file:
            st.markdown('<div class="image-viewer">', unsafe_allow_html=True)
            st.image(image, caption="📸 Ảnh X-quang gốc", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)