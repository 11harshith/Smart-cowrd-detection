#
# from ultralytics import YOLO
# import cv2
#
# # Load the model
# model = YOLO("best.pt")
#
# # Perform prediction
# results = model.predict(source="dataset/People/gettyimages-200244581-003-170667a.jpg", show=False, conf=0.6)
#
# # Display the image using OpenCV
# for result in results:
#     annotated_frame = result.plot()  # Get the annotated frame
#     cv2.imshow("YOLO Prediction", annotated_frame)
#
# # Wait until the user presses a key to close the window
# cv2.waitKey(0)  # 0 means wait indefinitely
# cv2.destroyAllWindows()



from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("models/yolo11n-pose.pt")

# Perform prediction
results = model.predict(source="1", show=True, conf=0.6)
