import streamlit as st
import openai
from gtts import gTTS
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. SETUP: Configuration & API Keys
# PM Note: Whisper Large V3 is the gold standard for multilingual STT in 2026 [5, 6]
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chained Audio Bot 2026", layout="wide")
st.title("🎙️ Conversational Audio Bot")
st.caption("Architecture: Chained Pipeline (Whisper + GPT-4o + gTTS)")

CHROMA_DIR = "./chroma_db"


def load_existing_vector_store():
    """
    Reload a previously persisted Chroma vector store from disk so that
    documents processed in earlier sessions remain queryable.
    Returns the vector store, or None if nothing has been persisted yet.
    """
    if "vector_store" in st.session_state:
        return st.session_state["vector_store"]

    # Only attempt to load if the persistence directory actually has data
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )
            st.session_state["vector_store"] = vector_store
            return vector_store
        except Exception as e:
            st.warning(f"Could not load existing knowledge base: {e}")
            return None
    return None

def process_and_store_document(uploaded_file):
    """
    Saves an uploaded PDF, splits it into chunks, embeds it, 
    and stores it in a local Chroma vector database.
    """
    # A. Save the uploaded file temporarily to disk
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        # B. Load the PDF document
        st.info("Reading document...")
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # C. Split the text into manageable chunks
        st.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # D. Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # E. Create and persist the local Chroma Vector Database
        st.info("Generating embeddings and saving to vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR # Directory where vectors are saved
        )
        
        st.success(f"Successfully processed {len(chunks)} text chunks!")
        return vector_store

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# 2. --- Streamlit UI Components ---
st.title("📄 Document Ingestion for Voice RAG")

# File uploader widget in the sidebar
with st.sidebar:
    st.header("Knowledge Base Setup")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            # Run the ingestion pipeline
            db = process_and_store_document(uploaded_file)
            if db:
                st.session_state["vector_store"] = db

# 3. SIDEBAR: TPM/PM Control Panel
with st.sidebar:
    st.header("⚙️ System Configuration")
    model_choice = st.selectbox("LLM Brain", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.slider("Grounding (Temperature)", 0.0, 1.0, 0.3) # Lower temp = fewer hallucinations [7]
    
    st.divider()
    st.header("📊 2026 Performance Metrics")
    st.info("Target TTFA: < 1.0s (Chained Architecture Baseline)") # Chained pipelines are slower than S2S [8, 9]
    st.metric("Whisper WER Target", "7.4%", delta="-2.1% vs V2") # Benchmark for Whisper Large V3 [10]

# Ensure any previously persisted knowledge base is available for retrieval
vector_store = load_existing_vector_store()
with st.sidebar:
    st.divider()
    st.header("📚 Knowledge Base Status")
    if vector_store is not None:
        st.success("RAG active — answers grounded in uploaded documents.")
    else:
        st.info("No documents indexed. Answers use general knowledge only.")

# 4. AUDIO INPUT: The "Ears" (ASR Stage)
audio_value = st.audio_input("Speak to the AI Assistant")

if audio_value:
    # A. ASR Stage: Whisper Transcription
    # PM Insight: Using Whisper Large V3 for 99+ language support [5, 6]
    start_time = time.time()
    with st.status("👂 Listening and Transcribing...", expanded=True):
        # Save temporary file for transcription
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.read())
        
        # Call OpenAI Whisper API (use context manager so the handle closes)
        with open("temp_input.wav", "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcript_response.text
        st.write(f"**User said:** {user_text}")

    # B. LLM Stage: GPT-4o Reasoning (RAG)
    # TPM Insight: Retrieved document context is injected here for grounding [11, 12]
    with st.status("🧠 Reasoning...", expanded=True):
        # B1. RETRIEVAL: pull the most relevant chunks from the vector store
        context = ""
        retrieved_docs = []
        if vector_store is not None:
            st.write("🔎 Searching knowledge base...")
            retrieved_docs = vector_store.similarity_search(user_text, k=4)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # B2. AUGMENTATION: build a grounded system prompt when context exists
        if context:
            system_prompt = (
                "You are a professional assistant. Answer the user's question using "
                "ONLY the context provided below. If the answer is not contained in "
                "the context, say you don't have that information in the documents. "
                "Be concise.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
            )
        else:
            system_prompt = (
                "You are a professional assistant. Be concise. "
                "No reference documents are available, so answer from general knowledge "
                "and note that the response is not grounded in any uploaded document."
            )

        # B3. GENERATION
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=temperature
        )
        ai_text = response.choices[0].message.content
        st.write(f"**AI Response:** {ai_text}")

        # Show the sources that grounded the answer
        if retrieved_docs:
            with st.expander(f"📎 Sources ({len(retrieved_docs)} chunks retrieved)"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} (page {page}):** {doc.page_content[:400]}...")

    # C. TTS Stage: gTTS Vocalization
    # TPM Note: gTTS provides high clarity but lower emotional nuance than Sarvam Bulbul V3 [13, 14]
    with st.status("🗣️ Synthesizing Voice...", expanded=True):
        tts = gTTS(text=ai_text, lang='en')
        tts.save("ai_response.mp3")
        
        # Calculate TTFA (Time to First Audio)
        ttfa = round(time.time() - start_time, 2)
        st.audio("ai_response.mp3", autoplay=True)

    # 5. FINAL PERFORMANCE REPORT
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Measured TTFA", f"{ttfa}s") # Tracks latency "tax" of chained models [1]
    col2.metric("Architecture", "Chained")
    col3.metric("Language Detection", "Automatic")
    
    if ttfa > 1.0:
        st.warning("⚠️ High Latency detected. Consider moving to Native S2S for sub-250ms performance.") # [4]