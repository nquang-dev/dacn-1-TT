import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile
import os
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
import numpy as np

from src.model import get_model

def load_all_styles():
    """Load tất cả file CSS"""
    css_files = [
        'styles/base.css',
        'styles/components.css', 
        'styles/single_analysis.css',
        'styles/batch_analysis.css',
        'styles/ai_assistant.css',
        'styles/user_guide.css'
    ]
    
    combined_css = ""
    for css_file in css_files:
        try:
            with open(css_file, 'r', encoding='utf-8') as f:
                combined_css += f.read() + "\n"
        except FileNotFoundError:
            st.warning(f"Không tìm thấy file CSS: {css_file}")
    
    st.markdown(f'<style>{combined_css}</style>', unsafe_allow_html=True)

@st.cache_resource
def load_model_cached():
    """Load model và các components cần thiết"""
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    last_conv_layer = model.layer4[-1]
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return model, last_conv_layer, preprocess

def register_fonts():
    """Đăng ký font cho PDF"""
    font_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'DejaVuSans.ttf'),
        os.path.join(os.path.dirname(__file__), '..', 'fonts', 'DejaVuSans.ttf'),
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
    
    return font_registered

def create_pdf_report(image, cam_image, prediction, prob_normal, prob_tb, process_time, filename=None):
    """Tạo báo cáo PDF"""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    vietnamese_font = 'DejaVuSans'
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        styles.add(ParagraphStyle(name='Vietnamese', fontName=vietnamese_font, fontSize=12))
    else:
        styles.add(ParagraphStyle(name='Vietnamese', fontName='Helvetica', fontSize=12))
    
    elements = []
    
    # Title
    title_style = styles["Heading1"]
    title_style.alignment = 1
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        title_style.fontName = vietnamese_font
    elements.append(Paragraph("KẾT QUẢ PHÂN TÍCH X-QUANG PHỔI", title_style))
    elements.append(Spacer(1, 20))
    
    # Date
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
    
    # Save images temporarily
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
    
    # Original image
    elements.append(Paragraph("Ảnh X-quang gốc:", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(img_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    # CAM image
    elements.append(Paragraph("Ảnh phân tích (CAM):", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(cam_path, width=400, height=300))
    elements.append(Spacer(1, 20))
    
    # Results
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
    
    # Data table
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
    
    # Note
    note_heading_style = styles["Heading3"]
    if vietnamese_font in pdfmetrics.getRegisteredFontNames():
        note_heading_style.fontName = vietnamese_font
    elements.append(Paragraph("Lưu ý:", note_heading_style))
    elements.append(Paragraph("Kết quả này chỉ mang tính chất tham khảo. Vui lòng tham khảo ý kiến của bác sĩ chuyên khoa để có chẩn đoán chính xác.", styles["Vietnamese"]))
    
    doc.build(elements)
    
    # Clean up
    os.unlink(img_path)
    os.unlink(cam_path)
    
    buffer.seek(0)
    return buffer
