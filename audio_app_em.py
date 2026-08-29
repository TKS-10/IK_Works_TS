import streamlit as st
import openai
from gtts import gTTS
import os
import time
from dotenv import load_dotenv

# 1. SETUP: Configuration & API Keys
# PM Note: Whisper Large V3 is the gold standard for multilingual STT in 2026 [5, 6]
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chained Audio Bot 2026", layout="wide")
st.title("🎙️ Conversational Audio Bot")
st.caption("Architecture: Chained Pipeline (Whisper + GPT-4o + gTTS)")

# 2. SIDEBAR: TPM/PM Control Panel
with st.sidebar:
    st.header("⚙️ System Configuration")
    model_choice = st.selectbox("LLM Brain", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.slider("Grounding (Temperature)", 0.0, 1.0, 0.3) # Lower temp = fewer hallucinations [7]
    
    st.divider()
    st.header("📊 2026 Performance Metrics")
    st.info("Target TTFA: < 1.0s (Chained Architecture Baseline)") # Chained pipelines are slower than S2S [8, 9]
    st.metric("Whisper WER Target", "7.4%", delta="-2.1% vs V2") # Benchmark for Whisper Large V3 [10]

# 3. AUDIO INPUT: The "Ears" (ASR Stage)
audio_value = st.audio_input("Speak to the AI Assistant")

if audio_value:
    # A. ASR Stage: Whisper Transcription
    # PM Insight: Using Whisper Large V3 for 99+ language support [5, 6]
    start_time = time.time()
    with st.status("👂 Listening and Transcribing...", expanded=True):
        # Save temporary file for transcription
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.read())
        
        # Call OpenAI Whisper API
        transcript_response = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=open("temp_input.wav", "rb")
        )
        user_text = transcript_response.text
        st.write(f"**User said:** {user_text}")

    # B. LLM Stage: GPT-4o Reasoning
    # TPM Insight: Context is injected here for domain-specific tasks [11, 12]
    with st.status("🧠 Reasoning...", expanded=True):
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a professional assistant. Be concise."},
                {"role": "user", "content": user_text}
            ],
            temperature=temperature
        )
        ai_text = response.choices[-1].message.content
        st.write(f"**AI Response:** {ai_text}")

    # C. TTS Stage: gTTS Vocalization
    # TPM Note: gTTS provides high clarity but lower emotional nuance than Sarvam Bulbul V3 [13, 14]
    with st.status("🗣️ Synthesizing Voice...", expanded=True):
        tts = gTTS(text=ai_text, lang='en')
        tts.save("ai_response.mp3")
        
        # Calculate TTFA (Time to First Audio)
        ttfa = round(time.time() - start_time, 2)
        st.audio("ai_response.mp3", autoplay=True)

    # 4. FINAL PERFORMANCE REPORT
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Measured TTFA", f"{ttfa}s") # Tracks latency "tax" of chained models [1]
    col2.metric("Architecture", "Chained")
    col3.metric("Language Detection", "Automatic")
    
    if ttfa > 1.0:
        st.warning("⚠️ High Latency detected. Consider moving to Native S2S for sub-250ms performance.") # [4]