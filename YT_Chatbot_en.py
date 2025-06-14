import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import tempfile
import subprocess
import webvtt

# Initialize Streamlit
st.set_page_config(page_title="🎬 YouTube Chat Assistant", layout="centered", page_icon="▶️")
st.title("🎥 YouTube Q&A Chatbot")

# Get Google API key from secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

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

# Improved subtitle fetching function with Streamlit Cloud compatibility
def get_transcript(video_id):
    # Create a temporary directory for Streamlit Cloud
    with tempfile.TemporaryDirectory() as temp_dir:
        subtitle_path = os.path.join(temp_dir, f"{video_id}.en.vtt")
        
        try:
            # Download subtitles
            result = subprocess.run([
                "yt-dlp",
                f"https://www.youtube.com/watch?v={video_id}",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--convert-subs", "vtt",
                "-o", os.path.join(temp_dir, f"{video_id}.%(ext)s")
            ], capture_output=True, text=True, check=True)
            
            # Check if subtitle file exists
            if not os.path.exists(subtitle_path):
                st.error(f"Subtitle file not found at {subtitle_path}")
                return None
                
            # Read and parse VTT file
            captions = []
            for caption in webvtt.read(subtitle_path):
                text = caption.text.strip()
                # Remove any formatting tags
                text = re.sub(r'<[^>]+>', '', text)
                captions.append(text)
                
            return " ".join(captions)
            
        except subprocess.CalledProcessError as e:
            st.error(f"yt-dlp error: {e.stderr}")
            return None
        except Exception as e:
            st.error(f"Error processing subtitles: {str(e)}")
            return None

@st.cache_resource
def load_or_create_index(video_id):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Use temp directory for Streamlit Cloud
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, f"faiss_index_{video_id}")
        
        if os.path.exists(index_path):
            vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
        else:
            text = get_transcript(video_id)
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
        st.error("❌ Could not process video subtitles. Please try another video.")
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

# Display chat history
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
            try:
                answer = qa_chain.invoke(user_input)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")