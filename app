import streamlit as st
import onnxruntime as ort
import numpy as np
import wave
import os

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
    <p>Upload a .wav audio file to transcribe it using an ONNX-based speech-to-text model.</p>
</div>
""", unsafe_allow_html=True)

# Load the ONNX model
@st.cache_resource
def load_model():
    try:
        session = ort.InferenceSession("models/speech_to_text_model.onnx")
        return session
    except Exception as e:
        st.error(f"Failed to load the ONNX model: {str(e)}")
        return None

session = load_model()
if session is None:
    st.stop()

# File uploader for audio
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Play the uploaded audio
    st.audio(temp_file_path, format="audio/wav")

    # Transcribe the audio
    try:
        with wave.open(temp_file_path, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        
        # Ensure the audio input is in the correct shape for the model
        # Note: You may need to adjust the input shape based on your model's requirements
        audio_input = np.expand_dims(audio, axis=0).astype(np.float32)
        
        # Run inference
        with st.spinner("Transcribing audio..."):
            output = session.run(None, {"audio_input": audio_input})[0]
        
        # Display the transcription
        st.markdown("### Transcription Result")
        st.markdown(f'<div class="transcription-box">{output}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Add a button to restart
if st.button("Transcribe Another Audio"):
    st.rerun()
