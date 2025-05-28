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
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as reportlab_colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
from pages.single_analysis import single_analysis_page
from pages.batch_analysis import batch_analysis_page
from pages.ai_assistant import ai_assistant_page
from pages.guide import guide_page

# Thiết lập trang
st.set_page_config(
    page_title="AI X-Ray Lung Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫁"
)

# Đăng ký font
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

# Tải CSS
with open('styles/base.css') as f:
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

# Tạo báo cáo PDF
def create_pdf_report(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename=None):
    # (Giữ nguyên hàm create_pdf_report từ file gốc)
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

# Tải mô hình
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

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">🔬</div>
        <h2>AI Lung Diagnostics</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🎯 Chọn chức năng:**")
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = "🔍 Phân tích đơn lẻ"
    if st.button("🔍 Phân tích đơn lẻ", key="btn_single", use_container_width=True):
        st.session_state.selected_function = "🔍 Phân tích đơn lẻ"
    if st.button("📊 Phân tích hàng loạt", key="btn_batch", use_container_width=True):
        st.session_state.selected_function = "📊 Phân tích hàng loạt"
    if st.button("🤖 Trợ lý AI", key="btn_ai", use_container_width=True):
        st.session_state.selected_function = "🤖 Trợ lý AI"
    if st.button("ℹ️ Hướng dẫn", key="btn_guide", use_container_width=True):
        st.session_state.selected_function = "ℹ️ Hướng dẫn"
    st.markdown(f"**Đang sử dụng:** {st.session_state.selected_function}")
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

# Điều hướng tab
selected_tab = st.session_state.selected_function
if selected_tab == "🔍 Phân tích đơn lẻ":
    single_analysis_page(model, preprocess, last_conv_layer, create_pdf_report)
elif selected_tab == "📊 Phân tích hàng loạt":
    batch_analysis_page(model, preprocess, last_conv_layer, create_pdf_report)
elif selected_tab == "🤖 Trợ lý AI":
    ai_assistant_page()
elif selected_tab == "ℹ️ Hướng dẫn":
    guide_page()