import streamlit as st
import numpy as np
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Speech-to-Text Transcription App", page_icon="🎙️", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        background-color: #1A3550;
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .transcription-box {
        background-color: #E6F4FA;
        color: #1A3550;
        padding: 16px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #2AB7CA 0%, #1A3550 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        transition: transform 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🎙️ Speech-to-Text Transcription App</h1>
    <p>Upload an audio file to transcribe it using a pre-trained speech-to-text model.</p>
</div>
""", unsafe_allow_html=True)

# Load the pre-trained model and processor
@st.cache_resource
def load_model():
    try:
        # Using a small pre-trained model for demonstration
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return processor, model
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        return None, None

processor, model = load_model()

def transcribe_audio(audio_bytes):
    try:
        # Convert bytes to numpy array
        audio_input, sample_rate = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
        
        # Process the audio
        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode the output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Transcribe button
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio..."):
            # Read the uploaded file
            audio_bytes = uploaded_file.read()
            
            # Transcribe
            transcription = transcribe_audio(audio_bytes)
            
            if transcription:
                st.markdown("### Transcription Result")
                st.markdown(f'<div class="transcription-box">{transcription}</div>', unsafe_allow_html=True)

# Add a button to restart
if st.button("Transcribe Another Audio"):
    st.rerun()
