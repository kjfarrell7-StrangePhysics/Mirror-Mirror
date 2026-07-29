import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Interactive AI Mirror", page_icon="🪞", layout="wide"
)

st.title("🪞 Interactive AI Mirror")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    thresh1 = st.slider("Canny Threshold 1", 0, 500, 100)
    thresh2 = st.slider("Canny Threshold 2", 0, 500, 200)

with col2:
    # Captures photo directly via browser without WebRTC socket blocks
    picture = st.camera_input("Take a snapshot to filter")

    if picture:
        # Convert image to OpenCV format
        bytes_data = picture.getvalue()
        cv_img = cv2.imdecode(
            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
        )

        # Apply Canny Edge Detection
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, thresh1, thresh2)

        # Display result
        st.image(
            edges, caption="Processed Image", use_container_width=True
        )
