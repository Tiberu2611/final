import streamlit as st
import os
import pickle
import faiss
import pytz
from datetime import datetime
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, models

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phòng Khám Đông Y AI", page_icon="🏥", layout="wide")

# --- KIỂM TRA ĐƯỜNG DẪN DỮ LIỆU ---
POSSIBLE_PATHS = ['Saved_Model/Saved_Model', 'Saved_Model', '.']
def find_data_path():
    for path in POSSIBLE_PATHS:
        if os.path.exists(os.path.join(path, "my_faiss.index")):
            return path
    return None

DATA_PATH = find_data_path()

# --- CẤU HÌNH API KEY (ẨN) ---
# Code sẽ tự động lấy từ mục Secrets của Streamlit Cloud
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
else:
    st.error("❌ Chưa cấu hình API Key trong Secrets. Vui lòng kiểm tra cài đặt trên Streamlit Cloud.")
    st.stop()

# --- CÁC HÀM HỖ TRỢ ---
def get_vietnam_time():
    tz_VN = pytz.timezone('Asia/Ho_Chi_Minh')
    return datetime.now(tz_VN).strftime("%d/%m/%Y - %H:%M")

@st.cache_resource
def load_embedding_model():
    with st.spinner("🔄 Đang tải mô hình ngôn ngữ..."):
        try:
            word_embedding_model = models.Transformer('vinai/phobert-base', max_seq_length=256)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            return SentenceTransformer(modules=[word_embedding_model, pooling_model])
        except:
            return SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

@st.cache_resource
def load_rag_system(folder_path):
    if folder_path:
        index_path = os.path.join(folder_path, "my_faiss.index")
        chunks_path = os.path.join(folder_path, "chunks.pkl")
        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            return index, chunks
        except Exception as e:
            st.error(f"Lỗi khi đọc file dữ liệu: {e}")
    return None, None

def retrieve_info(query, index, chunks, model, k=3):
    if index is None: return []
    q_emb = model.encode([query])[0].reshape(1, -1).astype('float32')
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

def generate_consultation(query, book_knowledge, patient_history):
    # Sử dụng tên model ổn định nhất để tránh lỗi 404
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    Bạn là một Bác sĩ Y học Cổ truyền chuyên nghiệp. 
    Thời gian hiện tại: {get_vietnam_time()}.
    DỮ LIỆU TỪ SÁCH: {book_knowledge}
    LỊCH SỬ KHÁM: {patient_history}
    CÂU HỎI BỆNH NHÂN: {query}
    ---
    CHỈ THỊ XỬ LÝ (TUÂN THỦ NGHIÊM NGẶT):

    BƯỚC 1: KIỂM TRA ĐỘ KHỚP THÔNG TIN (QUAN TRỌNG NHẤT)
    - Hãy đọc kỹ phần "KIẾN THỨC TRA CỨU TỪ SÁCH" ở trên.
    - Nếu câu hỏi của bệnh nhân chứa các từ khóa về:
        + Thuốc Tây y (Ví dụ: Panadol, Paracetamol, Kháng sinh, Aspirin...).
        + Bệnh danh hiện đại không có trong Đông y (Ví dụ: COVID-19, Ung thư giai đoạn cuối, Phẫu thuật, HIV...).
        + Hoặc nội dung trong phần "KIẾN THỨC TRA CỨU" hoàn toàn không liên quan đến câu hỏi.
    => HÀNH ĐỘNG: Trả lời ngắn gọn: "Xin lỗi, hệ thống dữ liệu Y học cổ truyền hiện tại không có thông tin về [Tên thuốc/Tên bệnh]. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa."
    => TUYỆT ĐỐI KHÔNG cố gắng lái sang tư vấn triệu chứng.
    => TUYỆT ĐỐI KHÔNG đưa ra lời khuyên thay thế. Dừng câu trả lời tại đây.

    BƯỚC 2: NẾU THÔNG TIN HỢP LỆ VÀ CÓ TRONG SÁCH ĐÔNG Y
    - Nếu bệnh nhân mô tả chung chung: Hãy hỏi ngược lại (Vấn chẩn) để làm rõ thể bệnh (Hư/Thực, Hàn/Nhiệt).
    - Nếu bệnh nhân mô tả rõ ràng: Đưa ra chẩn đoán và bài thuốc dựa trên "KIẾN THỨC TRA CỨU".
    - ĐỪNG đoán mò.
    - Hãy đóng vai người dẫn dắt, đưa ra 3-4 lựa chọn trắc nghiệm dựa trên các chứng bệnh trong sách để bệnh nhân chọn.
    - Ví dụ: "Đau bụng có nhiều thể. Bạn đau kiểu nào? 1. Đau âm ỉ (Hư hàn)? 2. Đau dữ dội (Thực tích)?"
    BƯỚC 3: TRÍCH DẪN
    - Mọi lời khuyên đưa ra phải dựa trên sách.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.header("⚙️ Quản lý")
    patient_id = st.text_input("Tên bệnh nhân", value="Khách")
    if st.button("Làm mới cuộc hội thoại"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    if DATA_PATH:
        st.success("✅ Hệ thống đã kết nối dữ liệu sách.")
    else:
        st.warning("⚠️ Đang chạy không có dữ liệu sách bổ trợ.")

if "messages" not in st.session_state:
    st.session_state.messages = []

embed_model = load_embedding_model()
faiss_index, all_chunks = load_rag_system(DATA_PATH)

st.title("🏥 Phòng Khám Đông Y AI")
st.caption(f"Trạng thái: Đang hoạt động | {get_vietnam_time()}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mô tả triệu chứng của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        if faiss_index:
            relevant = retrieve_info(prompt, faiss_index, all_chunks, embed_model)
            context = "\n".join(relevant)
        
        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
        
        with st.spinner("Bác sĩ đang xem hồ sơ..."):
            response = generate_consultation(prompt, context, history)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
