import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive Real-Time Mirror")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    # Toggle to flip video horizontally (Mirror View vs. True View)
    flip_view = st.checkbox("Flip View (Mirror Effect)", value=True)

    st.markdown("---")
    thresh1 = st.slider("Canny Threshold 1", 0, 500, 100)
    thresh2 = st.slider("Canny Threshold 2", 0, 500, 200)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Flip horizontally if checked
    if flip_view:
        img = cv2.flip(img, 1)

    # Canny Edge Processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh1, thresh2)
    img_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img_processed, format="bgr24")


RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
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
