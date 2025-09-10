import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Object Tracking 3D Class
class ObjectTracking3D:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.track_history = defaultdict(list)
        self.all_3d_points = defaultdict(list)
        self.frame_number = 0

        # Initialize 3D plot
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Frame Number')
        self.ax.set_title("3D Object Tracking")

    def track_objects(self):
        # Create a placeholder for live video
        live_frame_display = st.empty()
        plot_display = st.empty()

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                results = self.model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                annotated_frame = results[0].plot()

                # Update the tracking history and 3D points
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    self.all_3d_points[track_id].append((float(x), float(y), self.frame_number))

                # Display the updated frame in the live window
                _, buffer = cv2.imencode('.png', annotated_frame)
                img_bytes = buffer.tobytes()
                live_frame_display.image(img_bytes, channels="BGR", use_container_width=True)

                # Update the 3D plot
                self.ax.cla()  # Clear the previous plot
                for track_id, points in self.all_3d_points.items():
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z = [p[2] for p in points]
                    self.ax.plot(x, y, z, label=f"Track ID {track_id}")

                self.ax.set_xlabel('X Position')
                self.ax.set_ylabel('Y Position')
                self.ax.set_zlabel('Frame Number')
                self.ax.set_title("3D Object Tracking")
                self.ax.legend()

                # Display the updated 3D plot
                plot_display.pyplot(self.fig)

                self.frame_number += 1
            else:
                break

        self.cap.release()


def main():
    st.title("3D Object Tracking with YOLO")

    # Upload video file
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            video_path = temp_video_file.name

        # Path to the YOLO model (use your own model path here)
        model_path = "models/yolo11n-pose.onnx"

        # Initialize ObjectTracking3D class
        tracker = ObjectTracking3D(model_path, video_path)

        # Start tracking objects when the user clicks the button
        if st.button("Start Tracking"):
            tracker.track_objects()

            st.write("Tracking in progress...")
    else:
        st.write("Please upload a video to start tracking.")

if __name__ == "__main__":
    main()
