from fastapi import FastAPI, File, UploadFile
import requests
import shutil
import os

app = FastAPI()

STREAMLIT_URL = "https://bee-detection-app-vlnce3fpbya6pdi3xeacip.streamlit.app"

@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Send file to Streamlit app
    with open(temp_filename, "rb") as f:
        response = requests.post(f"{STREAMLIT_URL}/upload", files={"file": f})

    # Remove temp file
    os.remove(temp_filename)

    # Return JSON response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to process audio", "details": response.text}