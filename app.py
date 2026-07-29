import asyncio
import logging
import os

# Mute backend logging noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)


# Catch background network cleanup silently so it won't crash
def suppress_async_errors():
    try:
        loop = asyncio.get_event_loop()

        def custom_handler(loop, context):
            exception = context.get("exception")
            msg = str(context.get("message", ""))
            err_str = str(exception) if exception else ""
            if (
                "sendto" in err_str
                or "call_exception_handler" in err_str
                or "sendto" in msg
            ):
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(custom_handler)
    except Exception:
        pass


suppress_async_errors()

st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive Real-Time Mirror")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    thresh1 = st.slider("Canny Threshold 1", 0, 500, 100)
    thresh2 = st.slider("Canny Threshold 2", 0, 500, 200)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Edge detection filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh1, thresh2)
    img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img_processed, format="bgr24")


# Relay config to pass video through Streamlit Cloud firewall
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                ],
                "username": "openrelay",
                "credential": "openrelay",
            },
        ]
    }
)

with col2:
    webrtc_streamer(
        key="mirror-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
