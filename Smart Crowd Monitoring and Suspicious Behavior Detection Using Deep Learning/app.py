import streamlit as st
from Crowd_density import CrowdDetection
from suspicious_activity_detection import YoloObjectDetectionApp
from Office_Monitering import ObjectTracking3D
import tempfile

st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {"#ff5758"};
            color: {"#FFFFFF"};
        }}
        </style>
        """,
        unsafe_allow_html=True
)
def display_sidebar():
    st.sidebar.title("How It Works")
    st.sidebar.image("App_Images/se4.png")

    # Crowd Management Section
    st.sidebar.markdown("""
    ### Crowd Management
    The system uses YOLO (You Only Look Once) object detection to analyze real-time video feeds for crowd monitoring. 
    By identifying the density and movement patterns of people in specific areas, it helps in managing crowd flow, ensuring safety, and identifying potential bottlenecks or dangerous situations.
    """)

    # Crime Prevention Section
    st.sidebar.title("Crime Prevention")
    st.sidebar.image("App_Images/s22.png")
    st.sidebar.markdown("""
    The YOLO model is also trained to detect abnormal behaviors or suspicious activities in the crowd, such as aggressive actions, loitering, or unauthorized access to restricted areas. 
    Upon detecting such events, the system triggers alerts to security personnel for immediate action, preventing potential crimes in real-time.
    """)

    # Resource Movement in Office Spaces Section
    st.sidebar.title("Resource Movement in Office Spaces")
    st.sidebar.image("App_Images/se1.png")
    st.sidebar.markdown("""
    YOLO also tracks movement patterns of resources, including office equipment, personnel, and assets within the office environment. 
    By recognizing and logging the movement of these objects, it helps in resource management, ensuring the efficient use of office space, and preventing unauthorized movements.
    """)

    # Alert System Section
    st.sidebar.title("Alert System")
    st.sidebar.image("App_Images/se3.png")
    st.sidebar.markdown("""
    When suspicious behavior or unauthorized movements are detected, the system sends immediate alerts through Telegram or other messaging platforms. 
    This ensures that security teams or office managers are instantly informed, allowing them to take timely actions for resolution.
    """)


st.sidebar.title("Real-time Behavior Detection with YOLO")

display_sidebar()  # Display the sidebar content

    # Rest of your Streamlit application logic goes here...




# Constants
BOT_TOKEN = "7047754135:AAG8fFEA1lDVe21bQYYTozv3gb_wpf3-5hs"
CHAT_ID = "1893904443"
MODEL_PATH = "models/yolo11n.onnx"



# Streamlit UI setup
st.title("REAL-TIME BEHAVIOR DETECTION USING YOLO FOR CROWD MANAGEMENT, CRIME PREVENTION, AND RESOURCE MOVEMENT IN OFFICE SPACES WITH ALERT SYSTEM")
st.image("App_Images/coverpage.png")

# Dropdown menu for selecting functionality
option = st.sidebar.selectbox(
    "Select an option:",
    ["Choose the Option", "Crowd Detection", "Office Detection", "Suspicious Activity Detection"]
)

# Section for uploading a video (only displayed when Crowd Detection or Suspicious Activity Detection is selected)
if option == "Crowd Detection":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video file to a temporary location
        video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("Start Crowd Detection"):
            # Initialize and run crowd detection
            crowd_detector = CrowdDetection(BOT_TOKEN, CHAT_ID, MODEL_PATH, video_path)
            crowd_detector.detect_crowd()

elif option == "Office Detection":
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

elif option == "Suspicious Activity Detection":
    st.title("Suspicious Activity Detection")
    app = YoloObjectDetectionApp()
    app.run()
