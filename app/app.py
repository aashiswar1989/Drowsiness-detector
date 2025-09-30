import shutil
import json
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.background import P
from fastapi.responses import StreamingResponse
import uvicorn
from service import make_predictions
import gradio as gr
import requests
import sseclient
from DrowsinessDetector import logger



app = FastAPI()

@app.get('/')
def welcome():
    return "Welcome to the Drowsiness Detection API"

@app.post('/predict')
async def predict(video: UploadFile = File(...)):
    """
    Upload a video file and get drowsiness predictions per 30-frame sequence.
    - Accepts: .mp4, .avi, .mov files
    - Returns: Stream of predictions with confidence and label for each sequence
    """

    # 1. Validate file type
    if not video.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format.")

    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / video.filename

        # Saving uploaded file temporarily
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Yield predictions for each 30-frame sequence
        def stream_preds():
            try:
                for prediction in make_predictions(str(video_path)):
                    # print(json.dumps(prediction))
                    yield f"data: {json.dumps(prediction)}\n\n"  

            finally:
                # Clean up temporary file
                if video_path.exists():
                    video_path.unlink()

        return StreamingResponse(stream_preds(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
def run_uvicorn():
    uvicorn.run("app:app", host = '0.0.0.0', port = 8000)

FASTAPI_APP_URL = "http://127.0.0.1:8000/predict"


def stream_output(video_file):
    if video_file is None:
        logger.error("No video file provided.")
        return "No video upoaded. Please upload a video file."
    
    with open(video_file.name, "rb") as f:
        response = requests.post(FASTAPI_APP_URL, 
                                 files={"video": f},
                                 stream=True)
        
    if response.status_code != 200:
        logger.error(f"Request failed with status code {response.status_code}: {response.text}")
        return

    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.data:
            yield json.loads(event.data)

def gradio_interface():
    with gr.Blocks(title = "Driver Drowsiness Detection") as demo:
        gr.Markdown("Driver Drowsiness Detection using Facial Landmarks")

        with gr.Row():
            with gr.Column():
                video_input = gr.File(label = "Upload video in .mp4, .avi, .mov format", file_types = ['.mp4', '.avi', '.mov'])
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Reset")

            with gr.Column():
                prediction = gr.Textbox(label = "Live Prediction for a sequence of every 30 frames", 
                                        lines = 10,
                                        placeholder = "Predictions will be appear here...")
                
        submit_btn.click(fn = stream_output,
                         inputs = video_input,
                         outputs = prediction).then(fn= lambda : (None, "Video processed. Upload another video or reset."),
                                                    inputs = None,
                                                    outputs = [video_input, prediction])
        
        reset_btn.click(fn = lambda : (None, ""),
                        inputs = None,
                        outputs = [video_input, prediction])
        
        demo.launch()
        
if __name__ == "__main__":
    uvicorn_start = threading.Thread(target = run_uvicorn, daemon = True).start()
    gradio_interface()
