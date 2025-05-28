import streamlit as st
from pages.single_analysis import show_single_analysis
from pages.batch_analysis import show_batch_analysis
from pages.ai_assistant import show_ai_assistant
from pages.guide import show_guide
from utils.model_loader import load_model
import os

# Thiết lập trang với giao diện mới
st.set_page_config(
    page_title="AI X-Ray Lung Scanner", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="\U0001FAC1"
)

# Import CSS từ file riêng
def load_css():
    with open('styles/main.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Tải CSS tương ứng với tab hiện tại
    if "selected_function" in st.session_state:
        tab = st.session_state.selected_function
        if tab == "\U0001F50D Phân tích đơn lẻ":
            with open('styles/single_analysis.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        elif tab == "\U0001F4CA Phân tích hàng loạt":
            with open('styles/batch_analysis.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        elif tab == "\U0001F916 Trợ lý AI":
            with open('styles/ai_assistant.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        elif tab == "ℹ️ Hướng dẫn":
            with open('styles/guide.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# HEADER VỚI THIẾT KẾ CARD FLOATING
st.markdown("""
<div class="hero-section">
    <div class="floating-card">
        <div class="card-header">
            <div class="icon-wrapper">
                <div class="pulse-ring"></div>
                <div class="medical-icon">\U0001FAC1</div>
            </div>
            <div class="title-section">
                <h1 class="main-title">AI LUNG DIAGNOSTICS</h1>
                <p class="tagline">Hệ thống chẩn đoán thông minh cho X-quang phổi</p>
                <div class="feature-badges">
                    <span class="badge">\U0001F916 AI-Powered</span>
                    <span class="badge">⚡ Nhanh chóng</span>
                    <span class="badge">\U0001F3AF Chính xác</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tải mô hình với progress bar mới
try:
    with st.spinner("\U0001F504 Khởi tạo hệ thống AI..."):
        model, last_conv_layer = load_model()
    
    st.markdown("""
    <div class="success-notification">
        <div class="notification-icon">✨</div>
        <div class="notification-text">Hệ thống AI đã sẵn sàng hoạt động!</div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Không thể tải mô hình: {e}")
    st.stop()

# SIDEBAR MỚI VỚI NAVIGATION
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">\U0001F52C</div>
        <h2>AI Lung Diagnostics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Hoặc sử dụng buttons thay thế
    st.markdown("---")
    st.markdown("**\U0001F3AF Chọn chức năng:**")
    
    # Khởi tạo session state nếu chưa có
    if "selected_function" not in st.session_state:
        st.session_state.selected_function = "\U0001F50D Phân tích đơn lẻ"
    
    # Tạo buttons cho từng chức năng
    if st.button("\U0001F50D Phân tích đơn lẻ", key="btn_single", use_container_width=True):
        st.session_state.selected_function = "\U0001F50D Phân tích đơn lẻ"
    
    if st.button("\U0001F4CA Phân tích hàng loạt", key="btn_batch", use_container_width=True):
        st.session_state.selected_function = "\U0001F4CA Phân tích hàng loạt"
    
    if st.button("\U0001F916 Trợ lý AI", key="btn_ai", use_container_width=True):
        st.session_state.selected_function = "\U0001F916 Trợ lý AI"
    
    if st.button("ℹ️ Hướng dẫn", key="btn_guide", use_container_width=True):
        st.session_state.selected_function = "ℹ️ Hướng dẫn"
    
    # Hiển thị chức năng được chọn
    st.markdown(f"**Đang sử dụng:** {st.session_state.selected_function}")
    
    # Thống kê hệ thống
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-stats">
        <div class="stats-title">\U0001F4C8 Thống kê hệ thống</div>
        <div class="stat-item">
            <div class="stat-icon">\U0001F3AF</div>
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
            <div class="stat-icon">\U0001F52C</div>
            <div class="stat-info">
                <div class="stat-value">ResNet</div>
                <div class="stat-label">Mô hình AI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Hiển thị tab tương ứng
selected_tab = st.session_state.selected_function

if selected_tab == "\U0001F50D Phân tích đơn lẻ":
    show_single_analysis(model, last_conv_layer)
elif selected_tab == "\U0001F4CA Phân tích hàng loạt":
    show_batch_analysis(model, last_conv_layer)
elif selected_tab == "\U0001F916 Trợ lý AI":
    show_ai_assistant()
elif selected_tab == "ℹ️ Hướng dẫn":
    show_guide()
