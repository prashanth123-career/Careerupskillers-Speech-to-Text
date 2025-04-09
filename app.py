import streamlit as st
import numpy as np
from io import BytesIO
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf  # Alternative to librosa

# Set page config
st.set_page_config(page_title="Speech-to-Text", page_icon="🎙️")

# Custom CSS
st.markdown("""
<style>
    .header {background-color: #1A3550; color: white; padding: 20px; border-radius: 12px;}
    .transcription-box {background-color: #E6F4FA; padding: 16px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return processor, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def transcribe_audio(audio_bytes):
    try:
        # Use soundfile instead of librosa
        audio, sample_rate = sf.read(BytesIO(audio_bytes))
        if len(audio.shape) > 1:  # Convert stereo to mono
            audio = np.mean(audio, axis=1)
        
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None

# UI Components
st.markdown('<div class="header"><h1>🎙️ Speech-to-Text</h1></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg"])

processor, model = load_model()
if uploaded_file and processor and model:
    st.audio(uploaded_file)
    if st.button("Transcribe"):
        with st.spinner("Processing..."):
            result = transcribe_audio(uploaded_file.read())
            if result:
                st.markdown(f'<div class="transcription-box">{result}</div>', unsafe_allow_html=True)
