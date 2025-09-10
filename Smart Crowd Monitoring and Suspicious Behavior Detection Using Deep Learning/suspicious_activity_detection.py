import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np


class YoloObjectDetectionApp:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO("models/best.onnx")

    def detect_image_video(self, source):
        """Detect objects in an image or video."""
        results = self.model.predict(source=source, show=False, conf=0.6)
        # Annotate the image/video with detection boxes
        annotated_image = results[0].plot()  # This will draw the boxes on the image
        return annotated_image

    def handle_uploaded_file(self, uploaded_file):
        """Handle uploaded file (image or video)."""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            # For image files
            img = cv2.imdecode(file_bytes, 1)
            return self.detect_image_video(img)

        elif uploaded_file.type in ["video/mp4", "video/quicktime"]:
            # For video files
            video_file = cv2.VideoCapture(uploaded_file)
            return self.process_video(video_file)

    def process_video(self, video_file):
        """Process video frame by frame for detection."""
        stframe = st.empty()
        while video_file.isOpened():
            ret, frame = video_file.read()
            if not ret:
                break

            annotated_frame = self.detect_image_video(frame)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        video_file.release()

    def live_detection(self):
        """Live object detection using webcam."""
        cap = cv2.VideoCapture(1)  # Open the webcam
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to access webcam.")
                break

            annotated_frame = self.detect_image_video(frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        cap.release()

    def run(self):
        """Run the Streamlit app."""

        mode = st.selectbox("Choose Detection Mode", ["Upload Image/Video", "Live Detection"])

        if mode == "Upload Image/Video":
            uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov"])

            if uploaded_file is not None:
                annotated_output = self.handle_uploaded_file(uploaded_file)
                st.image(annotated_output, channels="BGR", caption="Processed Image/Video", use_container_width=True)

        elif mode == "Live Detection":
            self.live_detection()


# # Instantiate and run the app
# if __name__ == "__main__":
#     app = YoloObjectDetectionApp()
#     app.run()
