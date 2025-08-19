import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# ---- STREAMLIT UI ----
st.set_page_config(page_title="GenAI MVP", page_icon="🤖", layout="wide")
st.title("🧠 GenAI Architect Demo - RAG + Agentic AI + Guardrails")

st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])

query = st.text_input("💬 Ask your question:")

agent = st.radio("🤖 Choose Agent:", ["RAG Agent", "Guardrail Agent"])

# ---- FUNCTION: SIMPLE GUARDRAIL ----
def guardrail_check(user_input: str) -> bool:
    banned = ["ignore", "delete", "password", "hack"]
    return any(b in user_input.lower() for b in banned)

# ---- RAG PIPELINE ----
if openai_api_key and uploaded_file:
    if uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        text = ""
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    # Split + Embed
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(docs, embeddings)

    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    if query:
        if agent == "Guardrail Agent" and guardrail_check(query):
            st.error("🚫 Blocked by Guardrails: Unsafe or restricted request.")
        else:
            answer = qa.run(query)
            st.success(answer)
