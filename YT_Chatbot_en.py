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
import glob
import time

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
    if raw_input:  # Only show warning if input exists
        st.warning("⚠️ Please enter a valid YouTube URL or 11-character video ID.")
    st.stop()

def get_transcript(video_id):
    # First try YouTubeTranscriptApi
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        st.warning(f"⚠️ YouTubeTranscriptApi failed: {str(e)}. Trying yt-dlp...")
    
    # Fallback to yt-dlp
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Run yt-dlp command
            result = subprocess.run(
                [
                    "yt-dlp",
                    f"https://www.youtube.com/watch?v={video_id}",
                    "--write-auto-sub",
                    "--sub-lang", "en",
                    "--skip-download",
                    "--convert-subs", "vtt",
                    "--no-warnings",
                    "-o", os.path.join(temp_dir, "subtitle")
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find the generated subtitle file
            vtt_files = glob.glob(os.path.join(temp_dir, "subtitle*.vtt"))
            if not vtt_files:
                # Try alternative naming pattern
                vtt_files = glob.glob(os.path.join(temp_dir, f"*{video_id}*.vtt"))
            
            if not vtt_files:
                st.error(f"❌ No subtitle files found in temporary directory")
                return None
                
            # Use the first found VTT file
            subtitle_path = vtt_files[0]
            
            # Read and process the VTT content
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            
            # Simple parsing that works with VTT format
            lines = []
            for line in vtt_content.split('\n'):
                # Skip timestamps and metadata lines
                if '-->' in line or line.strip() == '' or line.startswith(('WEBVTT', 'NOTE', 'STYLE')):
                    continue
                # Remove any remaining HTML tags
                clean_line = re.sub(r'<[^>]+>', '', line).strip()
                if clean_line:
                    lines.append(clean_line)
                    
            return " ".join(lines)
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ yt-dlp error: {e.stderr if e.stderr else 'Unknown error'}")
            return None
        except Exception as e:
            st.error(f"❌ Error processing subtitles: {str(e)}")
            return None

@st.cache_resource(show_spinner=False)
def load_or_create_index(video_id):
    # Show loading status
    with st.spinner("🔍 Processing video subtitles..."):
        # Get transcript
        text = get_transcript(video_id)
        if not text:
            return None

        # Initialize embedding model
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Process text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        
        # Create and return vector store
        return FAISS.from_documents(chunks, embeddings_model)

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
You are an expert YouTube assistant. Answer questions based ONLY on the video transcript below.
If the question can't be answered using the transcript, respond: "I don't have information about that in the video."

Transcript excerpts:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
parser = StrOutputParser()
qa_chain = parallel_chain | prompt | model | parser

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display initial assistant message
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hi! I'm your YouTube assistant. Ask me anything about the video 📽️")

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
        with st.spinner("Analyzing video content..."):
            try:
                answer = qa_chain.invoke(user_input)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = "Sorry, I encountered an error. Please try a different question."
                st.error(f"❌ {error_msg}")
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})