import streamlit as st
import tempfile
import whisper

def transcribe_audio(file, model="tiny", language="en", prompt_words=None):
    whisper_model = whisper.load_model(model)
    if prompt_words:
        prompt_text = " ".join(prompt_words)
    st.write(f"Model is {'multilingual' if whisper_model.is_multilingual else 'English-only'}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    with st.spinner('Transcribing...'):
        if prompt_words:
            options = dict(language=language, beam_size=5, best_of=5, initial_prompt=prompt_text)
        else:
            options = dict(language=language, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        result = whisper_model.transcribe(tmp_file_path, **transcribe_options, fp16=False)

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



