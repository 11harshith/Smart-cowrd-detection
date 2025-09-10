import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import time
import numpy as np
from io import BytesIO

# Constants (no longer shown in UI)
BOT_TOKEN = "7047754135:AAG8fFEA1lDVe21bQYYTozv3gb_wpf3-5hs"
CHAT_ID = "1893904443"
MODEL_PATH = "models/yolo11n.onnx"

class CrowdDetection:
    def __init__(self, bot_token, chat_id, model_path, video_source):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        self.last_sent_time = time.time()
        self.overcrowd_sent = False

    def send_telegram_message(self, image, caption):
        """Send a message with an image to Telegram."""
        _, img_encoded = cv2.imencode('.png', image)
        img_bytes = img_encoded.tobytes()

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        payload = {'chat_id': self.chat_id, 'caption': caption}
        files = {'photo': ('image.png', img_bytes)}

        response = requests.post(url, data=payload, files=files)
        return response

    def detect_crowd(self):
        """Process video frames and detect crowds."""
        frame_placeholder = st.empty()  # Placeholder for video frames
        info_placeholder = st.empty()   # Placeholder for people count and label

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            results = self.model(frame)
            people = results[0].boxes.cls == 0  # Class 0 corresponds to 'person'
            boxes = results[0].boxes.xywh
            people_count = people.sum().item()

            # Determine box color and label based on crowd size
            if people_count >= 20:
                box_color = (0, 0, 255)
                label = "OverCrowd"
                if not self.overcrowd_sent:
                    self.send_telegram_message(frame, "Overcrowd detected!")
                    self.overcrowd_sent = True

                    caption = f"Overcrowd detected! View location: https://maps.app.goo.gl/F8J5oNPoyTyH6ZwBA\nPeople Count: {people_count}"
                    self.send_telegram_message(frame, caption)
            elif people_count >= 15:
                box_color = (0, 165, 255)
                label = "Crowd"
            elif people_count >= 10:
                box_color = (0, 255, 255)
                label = "Mini Crowd"
            elif people_count >= 5:
                box_color = (0, 255, 0)
                label = "Normal Crowd"
            else:
                box_color = (255, 255, 255)
                label = "No People"

            # Draw bounding boxes and display text
            for box in boxes[people]:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), box_color, 2)

                text = "Person"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x - w // 2, y - h // 2 - 20), (x - w // 2 + text_width, y - h // 2), (0, 0, 0), -1)
                cv2.putText(frame, text, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the count of people on the frame
            text_count = f"People Detected: {people_count}"
            (text_width, text_height), _ = cv2.getTextSize(text_count, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (20, 40 - text_height), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
            cv2.putText(frame, text_count, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display crowd density label
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (20, 80 - text_height), (20 + text_width, 80 + text_height), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

            # Convert frame to RGB for Streamlit and display it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update people count and label
            info_placeholder.text(f"People Detected: {people_count}\nCrowd Density: {label}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()

# Streamlit UI
def main():
    st.title("Crowd Detection with YOLO")

    # Upload video file
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video file
        video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("Start Crowd Detection"):
            # Use the constants for bot token, chat ID, and model path
            crowd_detector = CrowdDetection(BOT_TOKEN, CHAT_ID, MODEL_PATH, video_path)
            crowd_detector.detect_crowd()

if __name__ == "__main__":
    main()
