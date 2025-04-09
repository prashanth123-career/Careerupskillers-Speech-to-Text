import streamlit as st
import soundfile as sf
import numpy as np
from io import BytesIO
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Load model
@st.cache_resource
def load_model():
    return Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"), Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# App UI
st.title("🎙️ Speech-to-Text")
uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("Transcribe"):
        processor, model = load_model()
        audio_bytes = uploaded_file.read()
        
        try:
            audio, sample_rate = sf.read(BytesIO(audio_bytes))
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = np.mean(audio, axis=1)
                
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            transcription = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
            
            st.success("Transcription:")
            st.code(transcription)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
