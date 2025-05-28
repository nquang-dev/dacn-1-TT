import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

from src.model import get_model
from components.single_analysis import render_single_analysis
from components.batch_analysis import render_batch_analysis
from components.ai_assistant import render_ai_assistant
from components.user_guide import render_user_guide
from components.shared_utils import load_model_cached, load_all_styles

# Thiết lập trang
st.set_page_config(
    page_title="AI X-Ray Lung Scanner", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫁"
)

# Load tất cả CSS
load_all_styles()

# Load model
try:
    with st.spinner("🔄 Khởi tạo hệ thống AI..."):
        model, last_conv_layer, preprocess = load_model_cached()
    
    st.markdown("""
    <div class="success-notification">
        <div class="notification-icon">✨</div>
        <div class="notification-text">Hệ thống AI đã sẵn sàng hoạt động!</div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Không thể tải mô hình: {e}")
    st.stop()

# Hero Section
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

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-icon">🔬</div>
        <h2>AI Lung Diagnostics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
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
    
    # Sidebar stats (giữ nguyên)
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

# Main content routing
selected_tab = st.session_state.selected_function

if selected_tab == "🔍 Phân tích đơn lẻ":
    render_single_analysis(model, last_conv_layer, preprocess)
elif selected_tab == "📊 Phân tích hàng loạt":
    render_batch_analysis(model, last_conv_layer, preprocess)
elif selected_tab == "🤖 Trợ lý AI":
    render_ai_assistant()
elif selected_tab == "ℹ️ Hướng dẫn":
    render_user_guide()
