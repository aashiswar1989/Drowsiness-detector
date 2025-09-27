# app.py
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from service import make_predictions

# -------------------------------
# Create FastAPI instance
# -------------------------------
app = FastAPI(
    title="Drowsiness Detection API",
    description="Predicts drowsiness from a video using EAR/MAR LSTM model",
    version="1.0"
)

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Drowsiness Detection API"}

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Upload a video file and get drowsiness predictions per 30-frame sequence.
    - Accepts: .mp4 files
    - Returns: JSON with confidence and label for each sequence
    """

    # 1. Validate file type
    if not video.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")

    try:
        # 2. Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Use a unique filename to avoid conflicts
        unique_name = f"{uuid.uuid4()}_{video.filename}"
        video_path = temp_dir / unique_name

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 3. Run prediction using service.py
        predictions = make_predictions(str(video_path))

        # 4. Delete the file after processing (optional but recommended)
        video_path.unlink(missing_ok=True)

        # 5. Return JSON response
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # <package>.<file>:<FastAPI instance>
        host="127.0.0.1",
        port=8000,
        reload=True
    )