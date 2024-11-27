import gradio as gr
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64

API_URL = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
API_KEY = "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"
#API Used: meta/llama-3.2-90b-vision-instruct.

def extract_frames(video_path, fps=2): 
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    for frame_number in range(frame_count):
        ret, frame = video_capture.read()
        if ret and frame_number % int(frame_rate // fps) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    video_capture.release()
    return frames

def process_frames(frames, action):
    action_detected_count = 0
    stream = False

    for frame in frames:
        pil_image = Image.fromarray(frame)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        image_b64 = base64.b64encode(img_byte_arr).decode()

        payload = {
            "model": 'meta/llama-3.2-90b-vision-instruct',
            "messages": [
                {
                    "role": "user",
                    "content": f'What is in this image? <img src="data:image/png;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 256,
            "temperature": 0.3,  
            "top_p": 0.9,
            "stream": stream
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                recognition_result = response.json()
                action_detected = recognition_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if action.lower() in action_detected.lower():
                    action_detected_count += 1
            else:
                print(f"API Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {str(e)}")
        except Exception as e:
            print(f"General Exception: {str(e)}")

    return action_detected_count

def calculate_recognition_rate(action_detected_count, total_frames):
    return (action_detected_count / total_frames) * 100 if total_frames > 0 else 0

def process_videos(video1, video2, action):
    if not video1 or not video2:
        return "Please upload both videos.", None, None, None

    frames_video1 = extract_frames(video1)
    frames_video2 = extract_frames(video2)

    detected_frames_video1 = process_frames(frames_video1, action)
    detected_frames_video2 = process_frames(frames_video2, action)

    recognition_rate_video1 = calculate_recognition_rate(detected_frames_video1, len(frames_video1))
    recognition_rate_video2 = calculate_recognition_rate(detected_frames_video2, len(frames_video2))

    comparison = f"Video 1: {recognition_rate_video1:.2f}% action detected\n" \
                 f"Video 2: {recognition_rate_video2:.2f}% action detected\n" \
                 f"Comparison: {'Higher' if recognition_rate_video1 > recognition_rate_video2 else 'Lower' if recognition_rate_video1 < recognition_rate_video2 else 'Equal'} recognition rate for Video 1."

    return "Processing complete.", recognition_rate_video1, recognition_rate_video2, comparison

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Human Action Recognition using NVIDIA's Llama Model")
        with gr.Row():
            video1 = gr.Video(label="Upload Video 1")
            video2 = gr.Video(label="Upload Video 2")
        action = gr.Textbox(label="Action to Detect (e.g., jumping, running)")
        submit = gr.Button("Analyze Videos")

        result = gr.Textbox(label="Result", interactive=False)
        rate1 = gr.Textbox(label="Recognition Rate for Video 1", interactive=False)
        rate2 = gr.Textbox(label="Recognition Rate for Video 2", interactive=False)
        comparison = gr.Textbox(label="Comparison Summary", interactive=False)

        submit.click(process_videos, inputs=[video1, video2, action], outputs=[result, rate1, rate2, comparison])

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
