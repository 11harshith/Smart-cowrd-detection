from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO

model = YOLO("models/yolo11n-pose.onnx")
video_path = "Office Monitering/Office Background 2.mp4"
cap = cv2.VideoCapture(video_path)

# Dictionary to store tracking points for each track ID
track_history = defaultdict(list)

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Frame Number')
ax.set_title('3D Tracking of Objects')

# Store all the 3D points for visualization
all_3d_points = defaultdict(list)

frame_number = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Get tracking results
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate the frame with the detected objects and tracking
        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]

            # Append the new point to the track history
            track.append((float(x), float(y)))

            # Keep the last 30 points in history (optional limit to prevent excessive memory usage)
            if len(track) > 30:
                track.pop(0)

            # Draw the tracking line using polyline
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Store the 3D points (x, y, frame number)
            all_3d_points[track_id].append((float(x), float(y), frame_number))

        # Display the frame with tracking lines
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Update the 3D plot
        ax.cla()  # Clear the current axes
        for track_id, points in all_3d_points.items():
            points = np.array(points)
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], label=f"Track {track_id}")

        ax.legend()
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
