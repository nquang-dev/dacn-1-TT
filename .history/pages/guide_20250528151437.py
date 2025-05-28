import streamlit as st

def guide_page():
    st.markdown("""
    <div class="content-section">
        <div class="section-title">
            <span class="title-icon">ℹ️</span>
            <h2>Hướng dẫn sử dụng</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with open('styles/guide.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

    st.markdown("""
    <div class="footer-info">
        <div class="footer-content">
            <h4>🎓 Thông tin phát triển</h4>
            <p>Đồ án chuyên ngành - Ứng dụng AI trong y tế</p>
            <p>© 2025 - AI Lung Diagnostics System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)