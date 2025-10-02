import gradio as gr
import requests
import sseclient
import json
from DrowsinessDetector import logger

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
    gradio_interface()