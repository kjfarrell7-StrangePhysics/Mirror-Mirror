import asyncio
import logging
import os

# 1. Suppress C++ / MediaPipe stderr noise BEFORE importing cv2/mediapipe
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

# 2. Mute noisy Python WebRTC loggers
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)


# 3. Intercept and safely ignore aioice socket teardown race conditions
def install_asyncio_exception_handler():
    try:
        loop = asyncio.get_event_loop()

        def custom_handler(loop, context):
            exception = context.get("exception")
            msg = str(context.get("message", ""))
            err_str = str(exception) if exception else ""

            # Catch socket teardown exceptions when WebRTC connections close
            if (
                "sendto" in err_str
                or "call_exception_handler" in err_str
                or "sendto" in msg
            ):
                return  # Silently ignore socket teardown race conditions

            loop.default_exception_handler(context)

        loop.set_exception_handler(custom_handler)
    except Exception:
        pass


install_asyncio_exception_handler()

# -----------------------------------------------------------------------------
# Streamlit App & Layout Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive Real-Time Mirror")

col_ctrl, col_stream = st.columns([1, 2])

with col_ctrl:
    st.subheader("Settings")
    threshold1 = st.slider("Canny Threshold 1", 0, 500, 100)
    threshold2 = st.slider("Canny Threshold 2", 0, 500, 200)


# Thread-safe frame callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Image Processing Pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img_processed, format="bgr24")


# WebRTC Configuration with robust fallback STUN servers
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

with col_stream:
    webrtc_streamer(
        key="mirror-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
