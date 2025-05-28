import streamlit as st
import pandas as pd
import os

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

def show_ai_assistant():
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">\U0001F916</span>
            <h2>Trợ lý AI thông minh</h2>
        </div>
        <p class="section-desc">Đặt câu hỏi về bệnh lao phổi và cách sử dụng hệ thống</p>
    </div>
    """, unsafe_allow_html=True)
    
    qa_dict = load_chatbot_data()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    chat_col1, chat_col2 = st.columns([1, 1])
    
    # Phần còn lại của mã cho trợ lý AI...
    # (Mã còn lại được giữ nguyên từ app.py gốc)
