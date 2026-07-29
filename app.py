import logging
import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Interactive AI Mirror", page_icon="🪞", layout="wide")
st.title("🪞 Interactive Real-Time Mirror")

# Mute noisy WebRTC loggers without breaking the asyncio loop
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# 2. Controls & Interactive Parameters
# -----------------------------------------------------------------------------
col_ctrl, col_info = st.columns([1, 2])

with col_ctrl:
    st.subheader("Controls")
    threshold1 = st.slider("Canny Threshold 1", 0, 500, 100)
    threshold2 = st.slider("Canny Threshold 2", 0, 500, 200)

# -----------------------------------------------------------------------------
# 3. Frame Processing Callback (Thread-Safe)
# -----------------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert PyAV video frame to standard OpenCV BGR image
    img = frame.to_ndarray(format="bgr24")

    # Simple processing pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Return as an av.VideoFrame object
    return av.VideoFrame.from_ndarray(img_processed, format="bgr24")

# -----------------------------------------------------------------------------
# 4. Streamer Instance
# -----------------------------------------------------------------------------
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

webrtc_streamer(
    key="mirror-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
