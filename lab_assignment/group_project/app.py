import streamlit as st
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation, SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(
    page_title="DrugLaw AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-container h1 {
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        color: white !important;
    }
    .title-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Chat Bubble Design */
    .chat-bubble {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        line-height: 1.6;
        animation: fadeIn 0.4s ease-out;
    }
    .user-bubble {
        background-color: #e3f2fd;
        border-bottom-right-radius: 4px;
        margin-left: auto;
        color: #0d47a1;
        border: 1px solid #bbdefb;
    }
    .assistant-bubble {
        background-color: #ffffff;
        border-bottom-left-radius: 4px;
        margin-right: auto;
        color: #212121;
        border: 1px solid #eceff1;
    }
    
    /* Source metadata cards */
    .source-container {
        background: rgba(245, 247, 250, 0.85);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.8rem;
        border-left: 4px solid #1e3c72;
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Decorative elements */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        color: #fff;
        background-color: #28a745;
        margin-right: 0.5rem;
    }
    
    .badge-fallback {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# Query rephraser using history (Memory Engine)
def rephrase_query_with_history(query: str, chat_history: list) -> str:
    """
    Formulate query based on chat history to support follow-up questions.
    """
    if not chat_history:
        return query
        
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return query
        
    # Rephrase prompt
    history_context = ""
    for msg in chat_history[-4:]:  # last 2 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        history_context += f"{role}: {msg['content']}\n"
        
    prompt = f"""Dựa trên lịch sử hội thoại sau và câu hỏi mới nhất từ người dùng, hãy viết lại câu hỏi mới này thành một câu hỏi tìm kiếm độc lập bằng tiếng Việt.
Lưu ý: KHÔNG trả lời câu hỏi, chỉ trả về câu hỏi đã được viết lại. Nếu câu hỏi mới đã đầy đủ nghĩa độc lập, hãy trả về chính nó.

Lịch sử hội thoại:
{history_context}
Câu hỏi mới: {query}
Câu hỏi độc lập viết lại:"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        res.raise_for_status()
        rephrased = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return rephrased
    except Exception as e:
        return query


# Page Header
st.markdown("""
<div class="title-container">
    <h1>⚖️ Trợ Lý Tri Thức Pháp Luật & Tin Tức Ma Túy</h1>
    <p>Hệ thống RAG thông minh ứng dụng Hybrid Search, Reranking, Dynamic Diversification và Fallback PageIndex</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/law.png", width=96)
st.sidebar.markdown("### ⚙️ Cấu Hình RAG Pipeline")

# Select Model
model_choice = st.sidebar.selectbox(
    "1. Lựa chọn LLM Model",
    ["gemini-3.1-flash-lite (Recommended)", "gpt-4o-mini"],
    index=0
)

# Select Top-K
top_k_val = st.sidebar.slider(
    "2. Số lượng Chunks thu hồi (Top-K)",
    min_value=3,
    max_value=12,
    value=5
)

# Toggle Reranking
use_rerank = st.sidebar.checkbox(
    "3. Bật Reranking (Jina v2 / RRF)",
    value=True
)

# Temperature
temp = st.sidebar.slider(
    "4. Độ ngẫu nhiên (LLM Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

st.sidebar.markdown("---")

# Dynamic display of status and group info
st.sidebar.markdown("### 👥 Thành viên nhóm C2-C401")
members = [
    "Trần Hoàng Đạt - 2A202600807",
    "Nguyễn Văn Đoan - 2A202600795",
    "Lê Duy Hùng - 2A202600718",
    "Phạm Thị Tuyết Nga - 2A202600877",
    "Tạ Duy Xuân - 2A202600970"
]
for m in members:
    st.sidebar.text(f"• {m}")

# Main interface divided into Tabs
tab_chat, tab_docs, tab_eval = st.tabs(["💬 Chatbot Assistant", "📂 Tài liệu & Chunks", "📊 Bảng điểm Evaluation"])

with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 Chi tiết nguồn trích dẫn"):
                    for idx, src in enumerate(message["sources"], 1):
                        meta = src.get("metadata", {})
                        source_file = meta.get("source", "Không rõ nguồn")
                        score = src.get("score", 0.0)
                        content = src.get("content", "")
                        
                        st.markdown(f"**[{idx}] {source_file}** (Độ khớp: `{score:.3f}` | Loại: `{meta.get('type', 'news')}`)")
                        st.caption(content)
                        st.markdown("---")

    # User Input
    if prompt := st.chat_input("Nhập câu hỏi của bạn về luật ma túy hoặc tin showbiz..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất và tổng hợp câu trả lời..."):
                # 1. Rephrase query using history
                rephrased_q = rephrase_query_with_history(prompt, st.session_state.messages[:-1])
                if rephrased_q != prompt:
                    st.caption(f"🔄 Đã tối ưu truy vấn dựa trên ngữ cảnh: *\"{rephrased_q}\"*")
                
                # 2. Retrieve context
                chunks = retrieve(rephrased_q, top_k=top_k_val, use_reranking=use_rerank)
                
                # 3. Generate answer
                # Overwrite standard env vars if needed
                os.environ["TEMPERATURE"] = str(temp)
                
                res = generate_with_citation(rephrased_q, top_k=top_k_val)
                answer = res["answer"]
                sources = res["sources"]
                retrieval_src = res["retrieval_source"]
                
                # Display answer
                st.markdown(answer)
                
                # Show source badge
                if retrieval_src == "pageindex":
                    st.markdown('<span class="badge badge-fallback">PageIndex Fallback</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge">Hybrid Search</span>', unsafe_allow_html=True)
                
                # Display citations
                if sources:
                    with st.expander("🔍 Chi tiết nguồn trích dẫn"):
                        for idx, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            source_file = meta.get("source", "Không rõ nguồn")
                            score = src.get("score", 0.0)
                            content = src.get("content", "")
                            
                            st.markdown(f"**[{idx}] {source_file}** (Độ khớp: `{score:.3f}` | Loại: `{meta.get('type', 'news')}`)")
                            st.caption(content)
                            st.markdown("---")
                            
            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "retrieval_source": retrieval_src
            })

with tab_docs:
    st.markdown("### 📂 Danh sách Tài liệu & Chunks hiện có trong hệ thống")
    std_dir = PROJECT_ROOT / "data" / "standardized"
    
    if std_dir.exists():
        for sub in ["legal", "news"]:
            sub_path = std_dir / sub
            if sub_path.exists():
                st.subheader(f"Thư mục: `{sub}`")
                for md_file in sub_path.glob("*.md"):
                    with st.expander(f"📄 {md_file.name}"):
                        content = md_file.read_text(encoding="utf-8")
                        st.text_area("Nội dung gốc", content, height=200, key=str(md_file))
    else:
        st.warning("Thư mục dữ liệu chuẩn hóa `data/standardized` chưa được khởi tạo. Hãy chạy lại pipeline index trước.")

with tab_eval:
    st.markdown("### 📊 Kết quả kiểm thử Evaluation và A/B Testing (DeepEval)")
    eval_results_file = Path(__file__).parent / "evaluation" / "results.md"
    
    if eval_results_file.exists():
        results_content = eval_results_file.read_text(encoding="utf-8")
        st.markdown(results_content)
    else:
        st.info("Chưa tìm thấy báo cáo kết quả tự động. Vui lòng chạy file `group_project/evaluation/eval_pipeline.py` để tạo báo cáo.")
