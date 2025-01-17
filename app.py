import streamlit as st
import tempfile
import whisper

def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    with st.spinner('Transcribing...'):
        model = whisper.load_model("base")
        result = model.transcribe(tmp_file_path)
 
    return result["text"]

st.title("Speech to Text: Story Transcriptions")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    transcript = transcribe_audio(uploaded_file)
    st.write("Transcript:")
    st.write(transcript)
    st.download_button(
        label="Download Transcript",
        data=transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )

