import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.model import get_model
from src.visualization import apply_cam
import tempfile
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as reportlab_colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')

# Thiết lập trang với giao diện mới
st.set_page_config(
    page_title="AI X-Ray Lung Scanner", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫁"
)

# Đăng ký font hỗ trợ tiếng Việt
font_paths = [
    os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf'),
    os.path.join(os.path.dirname(__file__), 'fonts', 'DejaVuSans.ttf'),
    'DejaVuSans.ttf'
]

font_registered = False
for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            font_registered = True
            break
        except Exception as e:
            st.warning(f"Không thể đăng ký font từ {font_path}: {e}")

if not font_registered:
    st.warning("Không thể đăng ký font DejaVuSans. Sẽ sử dụng font mặc định.")

# Import CSS từ file riêng
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Tải mô hình
@st.cache_resource
def load_model():
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo báo cáo PDF (giữ nguyên)
def create_pdf_report(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename=None):
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    vietnamese_font = 'DejaVuSans'
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        styles.add(ParagraphStyle(name='Vietnamese', fontName=vietnamese_font, fontSize=12))
    else:
        styles.add(ParagraphStyle(name='Vietnamese', fontName='Helvetica', fontSize=12))
    
    elements = []
    
    title_style = styles["Heading1"]
    title_style.alignment = 1
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        title_style.fontName = vietnamese_font
    elements.append(Paragraph("KẾT QUẢ PHÂN TÍCH X-QUANG PHỔI", title_style))
    elements.append(Spacer(1, 20))
    
    date_style = styles["Normal"]
    date_style.alignment = 1
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        date_style.fontName = vietnamese_font
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    elements.append(Paragraph(f"Ngày giờ phân tích: {current_time}", date_style))
    elements.append(Spacer(1, 20))
    
    if filename:
        file_style = styles["Normal"]
        if vietnamese_font in pdfmetrics.getRegisteredFontNames():
            file_style.fontName = vietnamese_font
        elements.append(Paragraph(f"Tên file: {filename}", file_style))
        elements.append(Spacer(1, 10))
    
    img_path = tempfile.mktemp(suffix='.png')
    cam_path = tempfile.mktemp(suffix='.png')
    
    image.save(img_path)
    
    if isinstance(cam_image, np.ndarray):
        cam_image_pil = Image.fromarray(cam_image)
        cam_image_pil.save(cam_path)
    else:
        cam_image.save(cam_path)
    
    heading_style = styles["Heading2"]
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        heading_style.fontName = vietnamese_font
    
    elements.append(Paragraph("Ảnh X-quang gốc:", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(img_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Ảnh phân tích (CAM):", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(cam_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("KẾT QUẢ CHẨN ĐOÁN:", heading_style))
    elements.append(Spacer(1, 10))
    
    if prediction == 1:
        result_text = "PHÁT HIỆN DẤU HIỆU LAO PHỔI"
        result_color = reportlab_colors.red
    else:
        result_text = "KHÔNG PHÁT HIỆN DẤU HIỆU LAO PHỔI"
        result_color = reportlab_colors.green
    
    result_style = ParagraphStyle(
        name='ResultStyle',
        parent=styles["Heading2"],
        textColor=result_color,
        alignment=1
    )
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        result_style.fontName = vietnamese_font
    elements.append(Paragraph(result_text, result_style))
    elements.append(Spacer(1, 20))
    
    data = [
        ["Thông số", "Giá trị"],
        ["Xác suất bình thường", f"{prob_normal:.2%}"],
        ["Xác suất lao phổi", f"{prob_tb:.2%}"],
        ["Thời gian xử lý", f"{process_time:.2f} giây"]
    ]

    table = Table(data, colWidths=[200, 200])

    use_font = vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
    bold_font = vietnamese_font if vietnamese_font in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), reportlab_colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), reportlab_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), bold_font),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), reportlab_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.black),
        ('FONTNAME', (0, 1), (0, -1), use_font),
        ('FONTNAME', (1, 1), (1, -1), use_font),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    note_heading_style = styles["Heading3"]
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        note_heading_style.fontName = vietnamese_font
    elements.append(Paragraph("Lưu ý:", note_heading_style))
    elements.append(Paragraph("Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến của bác sĩ chuyên khoa để có chẩn đoán chính xác.", styles["Vietnamese"]))
    
    doc.build(elements)
    
    os.unlink(img_path)
    os.unlink(cam_path)
    
    buffer.seek(0)
    return buffer

# GIAO DIỆN MỚI - HEADER VỚI THIẾT KẾ CARD FLOATING
st.markdown("""
<div class="hero-section">
    <div class="floating-card">
        <div class="card-header">
            <div class="icon-wrapper">
                <div class="pulse-ring"></div>
                <div class="medical-icon">🫁</div>
            </div>
            <div class="title-section">
                <h1 class="main-title">AI LUNG DIAGNOSTICS</h1>
                <p class="tagline">Hệ thống chẩn đoán thông minh cho X-quang phổi</p>
                <div class="feature-badges">
                    <span class="badge">🤖 AI-Powered</span>
                    <span class="badge">⚡ Nhanh chóng</span>
                    <span class="badge">🎯 Chính xác</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tải mô hình với progress bar mới
try:
    with st.spinner("🔄 Khởi tạo hệ thống AI..."):
        model = load_model()
        last_conv_layer = model.layer4[-1]
    
    st.markdown("""
    <div class="success-notification">
        <div class="notification-icon">✨</div>
        <div class="notification-text">Hệ thống AI đã sẵn sàng hoạt động!</div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Không thể tải mô hình: {e}")
    st.stop()

# Tải dữ liệu chatbot
@st.cache_data
def load_chatbot_data():
    try:
        csv_path = "/home/nquang/C_CODE/2_DACN-1/tb_detection/data/chatbot_rules.csv"
        if not os.path.exists(csv_path):
            csv_path = "chatbot_rules.csv"
        
        df = pd.read_csv(csv_path)
        qa_dict = dict(zip(df["Question"], df["Answer"]))
        return qa_dict
    except Exception as e:
        st.error(f"Không thể tải dữ liệu chatbot: {e}")
        return {}

# SIDEBAR MỚI VỚI NAVIGATION
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">🔬</div>
        <h2>AI Lung Diagnostics</h2>
      
    </div>
    """, unsafe_allow_html=True)
    
  
    
    # Hoặc sử dụng buttons thay thế
    st.markdown("---")
    st.markdown("**🎯 Chọn chức năng:**")
    
    # Khởi tạo session state nếu chưa có
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = "🔍 Phân tích đơn lẻ"
    
    # Tạo buttons cho từng chức năng
    if st.button("🔍 Phân tích đơn lẻ", key="btn_single", use_container_width=True):
        st.session_state.selected_function = "🔍 Phân tích đơn lẻ"
    
    if st.button("📊 Phân tích hàng loạt", key="btn_batch", use_container_width=True):
        st.session_state.selected_function = "📊 Phân tích hàng loạt"
    
    if st.button("🤖 Trợ lý AI", key="btn_ai", use_container_width=True):
        st.session_state.selected_function = "🤖 Trợ lý AI"
    
    if st.button("ℹ️ Hướng dẫn", key="btn_guide", use_container_width=True):
        st.session_state.selected_function = "ℹ️ Hướng dẫn"
    
    # Hiển thị chức năng được chọn
    st.markdown(f"**Đang sử dụng:** {st.session_state.selected_function}")
    
    # Thống kê hệ thống
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-stats">
        <div class="stats-title">📈 Thống kê hệ thống</div>
        <div class="stat-item">
            <div class="stat-icon">🎯</div>
            <div class="stat-info">
                <div class="stat-value">99.2%</div>
                <div class="stat-label">Độ chính xác</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">⚡</div>
            <div class="stat-info">
                <div class="stat-value">< 3s</div>
                <div class="stat-label">Thời gian xử lý</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">🔬</div>
            <div class="stat-info">
                <div class="stat-value">ResNet</div>
                <div class="stat-label">Mô hình AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sử dụng session state thay vì biến selected_tab
selected_tab = st.session_state.selected_function


# MAIN CONTENT AREA
if selected_tab == "🔍 Phân tích đơn lẻ":
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">🔍</span>
            <h2>Phân tích ảnh X-quang đơn lẻ</h2>
        </div>
        <p class="section-desc">Tải lên một ảnh X-quang để phân tích và nhận kết quả chẩn đoán chi tiết</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout mới cho upload
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
            
            # Action buttons với thiết kế mới
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
                    
                    cam_image, prediction, prob_normal, prob_tb, process_time = apply_cam(
                        temp_file_path, model, preprocess, last_conv_layer)
                    
                    with preview_col:
                        # Hiển thị kết quả với thiết kế card mới
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
                        
                        # Hiển thị ảnh CAM
                        st.markdown('<div class="image-viewer">', unsafe_allow_html=True)
                        st.image(cam_image, caption="🎯 Vùng phân tích CAM", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Biểu đồ xác suất mới
                        fig, ax = plt.subplots(figsize=(8, 4))
                        categories = ['Bình thường', 'Lao phổi']
                        values = [prob_normal, prob_tb]
                        colors = ['#00D4AA', '#FF6B6B']
                        
                        bars = ax.barh(categories, values, color=colors)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Xác suất')
                        ax.set_title('📊 Phân tích xác suất', fontweight='bold')
                        
                        for i, (bar, value) in enumerate(zip(bars, values)):
                            ax.text(value + 0.02, i, f'{value:.1%}', 
                                   va='center', fontweight='bold')
                        
                        st.pyplot(fig)
                        
                        # Thông tin chi tiết
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
                        
                        # PDF download
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

elif selected_tab == "📊 Phân tích hàng loạt":
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">📊</span>
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
            
            # # Biểu đồ tròn
            # if total > 0:
            #     fig, ax = plt.subplots(figsize=(6, 6))
            #     labels = ['Bình thường', 'Cần chú ý']
            #     sizes = [normal_count, tb_count]
            #     colors = ['#00D4AA', '#FF6B6B']
            #     explode = (0.05, 0.05)
                
            #     wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
            #                                      colors=colors, autopct='%1.1f%%',
            #                                      shadow=True, startangle=90)
            #     ax.set_title('📈 Phân bố kết quả', fontweight='bold', fontsize=14)
                
            #     st.pyplot(fig)
            
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

elif selected_tab == "🤖 Trợ lý AI":
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">🤖</span>
            <h2>Trợ lý AI thông minh</h2>
        </div>
        <p class="section-desc">Đặt câu hỏi về bệnh lao phổi và cách sử dụng hệ thống</p>
    </div>
    """, unsafe_allow_html=True)
    
    qa_dict = load_chatbot_data()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    chat_col1, chat_col2 = st.columns([1, 1])
    
    with chat_col1:
        st.markdown("""
        <div class="chat-sidebar">
            <div class="chat-header">
                <span class="chat-icon">💬</span>
                <h3>Câu hỏi thường gặp</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Xóa lịch sử", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown('<div class="question-cards">', unsafe_allow_html=True)
        for i, question in enumerate(qa_dict.keys()):
            if st.button(f"❓ {question}", key=f"q_{i}", use_container_width=True):
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": qa_dict[question]
                })
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chat_col2:
        st.markdown("""
        <div class="chat-area">
            <div class="chat-header">
                <span class="chat-icon">💭</span>
                <h3>Cuộc trò chuyện</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="empty-chat-state">
                <div class="empty-icon">🤖</div>
                <h4>Chào bạn!</h4>
                <p>Hãy chọn một câu hỏi từ danh sách bên trái để bắt đầu trò chuyện.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-bubble user">
                    <div class="bubble-avatar">👤</div>
                    <div class="bubble-content">
                        <div class="bubble-text">{chat["question"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-bubble bot">
                    <div class="bubble-avatar">🤖</div>
                    <div class="bubble-content">
                        <div class="bubble-text">{chat["answer"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif selected_tab == "ℹ️ Hướng dẫn":
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">ℹ️</span>
            <h2>Hướng dẫn sử dụng</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tạo accordion-style info sections
    info_sections = [
        {
            "title": "🎯 Giới thiệu hệ thống",
            "content": """
            **AI Lung Diagnostics** là hệ thống chẩn đoán thông minh sử dụng trí tuệ nhân tạo 
            để phát hiện sớm các dấu hiệu bệnh lao phổi qua ảnh X-quang ngực.
            
            Hệ thống được phát triển với công nghệ học sâu tiên tiến, 
            giúp hỗ trợ các bác sĩ trong quá trình chẩn đoán.
            """
        },
        {
            "title": "📋 Hướng dẫn sử dụng",
            "content": """
            **Bước 1:** Chọn chức năng phù hợp từ sidebar bên trái
            - 🔍 Phân tích đơn lẻ: Cho một ảnh X-quang
            - 📊 Phân tích hàng loạt: Cho nhiều ảnh cùng lúc
            
            **Bước 2:** Tải lên ảnh X-quang (JPG, JPEG, PNG)
            
            **Bước 3:** Nhấn "Bắt đầu phân tích"
            
            **Bước 4:** Xem kết quả và tải báo cáo PDF
            """
        },
        {
            "title": "🧠 Về công nghệ AI",
            "content": """
            **Mô hình học sâu:** Sử dụng ResNet được huấn luyện trên hàng nghìn ảnh X-quang
            
            **CAM (Class Activation Map):** Trực quan hóa vùng nghi ngờ trên ảnh
            
            **Độ chính xác:** 99.2% trên tập dữ liệu kiểm thử
            
            **Thời gian xử lý:** Dưới 3 giây cho mỗi ảnh
            """
        }
    ]
    
    for section in info_sections:
        with st.expander(section["title"], expanded=False):
            st.markdown(section["content"])
    
    # Warning section
    st.markdown("""
    <div class="warning-panel">
        <div class="warning-header">
            <span class="warning-icon">⚠️</span>
            <h3>Lưu ý quan trọng</h3>
        </div>
        <div class="warning-content">
            <p><strong>Kết quả từ hệ thống này chỉ mang tính chất tham khảo và hỗ trợ.</strong></p>
            <p>Không thay thế chẩn đoán của bác sĩ chuyên khoa. 
            Vui lòng tham khảo ý kiến bác sĩ để có chẩn đoán chính xác.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer info
    st.markdown("""
    <div class="footer-info">
        <div class="footer-content">
            <h4>🎓 Thông tin phát triển</h4>
            <p>Đồ án chuyên ngành - Ứng dụng AI trong y tế</p>
            <p>© 2025 - AI Lung Diagnostics System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

