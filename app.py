import streamlit as st
from rag_pipeline import *

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Custom UI
st.markdown("""
<style>
.user-msg {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: right;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Construction RAG Chatbot")

# Setup
@st.cache_resource
def setup_rag():
    text = load_all_pdfs("data/")
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    index = build_index(embeddings)
    return chunks, index

chunks, index = setup_rag()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
query = st.text_input("Ask your question:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieval
    retrieved_chunks = retrieve(query, index, chunks)

    # Generation
    answer = generate_answer(query, retrieved_chunks)

    # Transparency
    context_text = "\n\n".join(retrieved_chunks)

    full_response = f"""
📄 Retrieved Context:
{context_text}

💡 Final Answer:
{answer}
"""

    st.session_state.messages.append({"role": "bot", "content": full_response})

    st.rerun()
