import shutil
import json
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.background import P
from fastapi.responses import StreamingResponse
import uvicorn
from service import make_predictions

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
            
if __name__ == "__main__":
    uvicorn.run("app:app", host = '0.0.0.0', port = 8000)
