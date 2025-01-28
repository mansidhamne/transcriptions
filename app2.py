import streamlit as st
import tempfile
import whisper
import os

# Add caching for the model loading
@st.cache_resource
def load_whisper_model(model_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["XDG_CACHE_HOME"] = temp_dir
        return whisper.load_model(model_name)

def transcribe_audio(file, model="tiny", language="en", prompt_words=None):
    # Use the cached model loader instead of loading directly
    whisper_model = load_whisper_model(model)
    
    if prompt_words:
        prompt_text = " ".join(prompt_words)
    
    st.write(f"Model is {'multilingual' if whisper_model.is_multilingual else 'English-only'}")
    
    # Use a context manager for the temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()  # Ensure all data is written
        
        with st.spinner('Transcribing...'):
            if prompt_words:
                options = dict(language=language, beam_size=5, best_of=5, initial_prompt=prompt_text)
            else:
                options = dict(language=language, beam_size=5, best_of=5)
            transcribe_options = dict(task="transcribe", **options)
            result = whisper_model.transcribe(tmp_file.name, **transcribe_options, fp16=False)

    return result["text"]

st.title("Speech to Text: Story Transcriptions")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

model = st.selectbox("Choose Whisper Model", ["tiny", "base", "medium"])
language = st.selectbox("Choose Language", ["en", "hi"])   
words_input = st.text_area(
    "Enter each prompt word on a new line:",
    height=200,
    placeholder="Type each word and press 'Enter' to move to the next line."
)
prompt_words = [word.strip() for word in words_input.split("\n") if word.strip()]
if prompt_words:
    st.write(prompt_words)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    transcript = transcribe_audio(uploaded_file, model, language, prompt_words)
    st.write("Transcript:")
    st.write(transcript)
    st.download_button(
        label="Download Transcript",
        data=transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )



