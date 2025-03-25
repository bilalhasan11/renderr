import os
import requests
import shutil
from fastapi import FastAPI, UploadFile, File

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

# Ensure FastAPI runs on the correct port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
