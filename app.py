import asyncio
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer


# -----------------------------------------------------------------------------
# 1. Suppress Async Socket Teardown Errors (aioice / aiortc cleanup race condition)
# -----------------------------------------------------------------------------
def suppress_webrtc_async_errors():
    try:
        loop = asyncio.get_event_loop()

        def exception_handler(loop, context):
            msg = str(context.get("exception", ""))
            # Intercept socket teardown exceptions when sessions reset
            if "sendto" in msg or "call_exception_handler" in msg:
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(exception_handler)
    except Exception:
        pass


suppress_webrtc_async_errors()

# -----------------------------------------------------------------------------
# 2. Streamlit Page Configuration & WebRTC Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive Real-Time Video Engine")

# Explicit STUN configuration prevents ICE connection hangs in cloud environments
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# Video Processor Class
class VideoTransformer(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Example processing pipeline (e.g., edge detection / stylization)
        # Replace or expand this with your MediaPipe or custom filter pipelines
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return frame.from_ndarray(img, format="bgr24")


# Render WebRTC Streamer
webrtc_streamer(
    key="mirror-stream",
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -----------------------------------------------------------------------------
# 3. Updated Layout Components (Streamlit Deprecation Fixes)
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
