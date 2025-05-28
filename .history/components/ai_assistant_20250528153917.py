import streamlit as st
import pandas as pd
import os

def load_chatbot_data():
    """Load dữ liệu chatbot"""
    try:
        csv_path = "/home/nquang/C_CODE/2_DACN-1/tb_detection/data/chatbot_rules.csv"
        if not os.path.exists(csv_path):
            csv_path = "data/chatbot_rules.csv"
        
        df = pd.read_csv(csv_path)
        qa_dict = dict(zip(df["Question"], df["Answer"]))
        return qa_dict
    except Exception as e:
        st.error(f"Không thể tải dữ liệu chatbot: {e}")
        return {}

def render_ai_assistant():
    """Render tab trợ lý AI"""
    
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
