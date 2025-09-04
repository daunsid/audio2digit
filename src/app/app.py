import streamlit as st
from ai.predict import predict_digit


st.title("🎙️ Spoken Digit Recognition")
st.write("Upload an audio file or record using your microphone to predict the spoken digit (0-9).")

# --- User options ---
option = st.radio("Choose input method:", ("Upload audio file", "Record from microphone"))

audio_bytes = None
audio_path = None
MODEL_PATH="model_checkpoint/digit_cnn.pth"


if option == "Upload audio file":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")
elif option == "Record from microphone":
    audio_bytes = st.audio_input("Record your digit")
    if audio_bytes is not None:
        audio_bytes = audio_bytes.read()
        st.audio(audio_bytes, format="audio/wav")

# --- Run prediction ---
if st.button("Predict") and (audio_bytes or audio_path):
    with st.spinner("Analyzing audio..."):
        try:
            pred = predict_digit(
                model_path=MODEL_PATH,
                device="cpu",
                audio_path=audio_path,
                audio_byte=audio_bytes
            )
            st.success(f"🎉 Predicted digit: **{pred}**")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
