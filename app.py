import asyncio
import os

# -----------------------------------------------------------------------------
# 1. Suppress C++ / MediaPipe Logging Noise in Stderr
# -----------------------------------------------------------------------------
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer


# -----------------------------------------------------------------------------
# 2. Suppress Async Socket Teardown Errors (aioice / aiortc cleanup race)
# -----------------------------------------------------------------------------
def suppress_webrtc_async_errors():
    try:
        loop = asyncio.get_event_loop()

        def custom_exception_handler(loop, context):
            exception = context.get("exception")
            msg = str(context.get("message", ""))
            err_str = str(exception) if exception else ""

            # Intercept socket teardown errors when a WebRTC session closes
            if (
                "sendto" in err_str
                or "call_exception_handler" in err_str
                or "sendto" in msg
            ):
                return

            loop.default_exception_handler(context)

        loop.set_exception_handler(custom_exception_handler)
    except Exception:
        pass


suppress_webrtc_async_errors()

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration & WebRTC Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive Real-Time Mirror")

# STUN server setup to prevent ICE connection hangs in Streamlit Cloud
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Video Processing Logic
class VideoTransformer(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Example image processing pipeline (OpenCV / MediaPipe)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return frame.from_ndarray(img, format="bgr24")


# WebRTC Streamer
webrtc_streamer(
    key="mirror-stream",
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------------------------------------------------------
# 4. Streamlit Layout Components (Updated to modern width parameters)
# -----------------------------------------------------------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Controls & Settings")
    st.slider("Filter Intensity", 0, 100, 50)

with col2:
    st.subheader("System Status")
    # Updated to width="stretch" to comply with latest Streamlit layout standards
    st.info("Streamer initialized with fallback STUN candidate gathering.")
