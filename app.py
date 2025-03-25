import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests
import json

# Load the model (assumes model is in the project folder)
MODEL_PATH = "mobilenet_best_model.keras"
model = load_model(MODEL_PATH)

def create_mel_spectrogram(audio_segment, sr):
    try:
        spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        plt.figure(figsize=(2, 2), dpi=100)
        plt.axis('off')
        plt.imshow(spectrogram_db, aspect='auto', cmap='magma', origin='lower')
        plt.tight_layout(pad=0)
        temp_image_path = "temp_spectrogram.png"
        plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        img = Image.open(temp_image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        os.remove(temp_image_path)
        return img_array
    except Exception as e:
        st.error(f"Error creating spectrogram: {e}")
        return None

def predict_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 10:
            return {"error": "Audio file must be longer than 10 seconds"}
        bee_count = 0
        total_segments = 0
        segment_start = 0
        while segment_start < duration:
            segment_end = min(segment_start + 10, duration)
            if segment_end - segment_start < 10 and segment_start > 0:
                segment_start = max(0, duration - 10)
                segment_end = duration
            audio_segment = y[int(segment_start * sr):int(segment_end * sr)]
            spectrogram = create_mel_spectrogram(audio_segment, sr)
            if spectrogram is not None:
                spectrogram = np.expand_dims(spectrogram, axis=0)
                prediction = model.predict(spectrogram)
                probability = prediction[0][0]
                if probability <= 0.2:
                    bee_count += 1
                total_segments += 1
            segment_start += 10
        if total_segments > 0:
            bee_percentage = (bee_count / total_segments) * 100
            result = "Bee" if bee_percentage >= 70 else "Not Bee"
            return {"result": result, "bee_percentage": bee_percentage, "segments_analyzed": total_segments}
        else:
            return {"result": "No segments processed"}
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {"error": str(e)}

def main():
    st.title("Bee Detection App")
    st.write("Upload an audio file or provide a URL to detect if it contains bee sounds!")

    # Option selection
    option = st.radio("Choose an input method:", ("Upload Audio File", "Enter Audio URL"))

    # File upload option
    if option == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            with st.spinner("Processing your audio..."):
                temp_path = "temp_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                result = predict_audio(temp_path)
                os.remove(temp_path)

                # Display result
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Result: *{result['result']}*")
                    st.write(f"Bee Sound Percentage: {result['bee_percentage']:.2f}%")
                    st.write(f"Segments Analyzed: {result['segments_analyzed']}")
                    st.json(result)  # Show JSON for API compatibility

    # URL input option
    elif option == "Enter Audio URL":
        audio_url = st.text_input("Enter the URL of an audio file:")
        if st.button("Analyze URL") and audio_url:
            with st.spinner("Downloading and processing audio..."):
                try:
                    response = requests.get(audio_url, timeout=10)
                    temp_path = "temp_audio.wav"
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
                    result = predict_audio(temp_path)
                    os.remove(temp_path)

                    # Display result
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Result: *{result['result']}*")
                        st.write(f"Bee Sound Percentage: {result['bee_percentage']:.2f}%")
                        st.write(f"Segments Analyzed: {result['segments_analyzed']}")
                        st.json(result)  # Show JSON for API compatibility
                except Exception as e:
                    st.error(f"Failed to process audio URL: {str(e)}")

    # API support via query parameters (for programmatic use)
    query_params = st.query_params
    if "audio_url" in query_params:
        audio_url = query_params["audio_url"]
        with st.spinner("Processing audio from URL..."):
            try:
                response = requests.get(audio_url, timeout=10)
                temp_path = "temp_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                result = predict_audio(temp_path)
                os.remove(temp_path)
                st.json(result)  # Return JSON for API calls
            except Exception as e:
                st.json({"error": f"Failed to process audio URL: {str(e)}"})

if __name__ == "__main__":
    main()