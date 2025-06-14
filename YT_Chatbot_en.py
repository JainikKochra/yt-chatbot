import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
from dotenv import load_dotenv

import subprocess
import webvtt

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="🎬 YouTube Chat Assistant", layout="centered",page_icon="▶️")
st.title("🎥 YouTube Q&A Chatbot")


# Single input field for YouTube URL or ID
raw_input = st.text_input("Enter YouTube URL or ID", key="youtube_input")

# Extract video ID using regex
match = re.search(r"(?:v=|be/)?([\w-]{11})", raw_input)
if match:
    video_id = match.group(1)
    st.success(f"✅ Extracted Video ID: {video_id}")
else:
    st.warning("⚠️ Please enter a valid YouTube URL or 11-character video ID.")
    st.stop()

# Fetch and parse subtitles using yt-dlp
def get_transcript_via_ytdlp(video_id):
    subtitle_dir = "./projcts/subtitles"
    os.makedirs(subtitle_dir, exist_ok=True)
    subtitle_path = os.path.join(subtitle_dir, f"{video_id}.en.vtt")

    if not os.path.exists(subtitle_path):
        try:
            subprocess.run([
                "yt-dlp",
                f"https://www.youtube.com/watch?v={video_id}",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "-o", os.path.join(subtitle_dir, f"{video_id}.%(ext)s")
            ], check=True)
        except subprocess.CalledProcessError:
            st.error("❌ Failed to download subtitles using yt-dlp.")
            return None

    try:
        captions = [caption.text.strip() for caption in webvtt.read(subtitle_path)]
        return " ".join(captions)
    except Exception:
        st.error("❌ Failed to parse the subtitle file.")
        return None

@st.cache_resource
def load_or_create_index(video_id):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index_path = f"./projcts/faiss_index"

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        text = get_transcript_via_ytdlp(video_id)
        if not text:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        vector_store.save_local(index_path)

    return vector_store

# Load vector store
if video_id:
    vector_store = load_or_create_index(video_id)
    if not vector_store:
        st.stop()
else:
    st.stop()

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Chain setup
def format_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_context),
    "question": RunnablePassthrough()
})

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"]
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
parser = StrOutputParser()
qa_chain = parallel_chain | prompt | model | parser

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display initial assistant message
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hi! Ask me anything about the video 📽️")

# Display chat history excluding the initial assistant message
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about the video...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(user_input)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

