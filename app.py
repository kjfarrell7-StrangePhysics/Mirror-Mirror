import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp
import replicate
import io

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Mirror App", page_icon="🪞", layout="centered")
st.title("🪞 AI Mirror")

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("🪞 Viewport Settings")
mode = st.sidebar.radio(
    "Mirror Mode:",
    ["Standard Mirror (Flipped)", "True View (How others see you)"]
)

st.sidebar.header("✨ Generative AI Transformations")
ai_mode = st.sidebar.selectbox("AI Transformation:", ["None", "Age Transformation", "Gender Swap"])

target_age = 70
gender_target = "masculine"

if ai_mode == "Age Transformation":
    target_age = st.sidebar.slider("Target Age:", 10, 90, 70)
elif ai_mode == "Gender Swap":
    gender_target = st.sidebar.radio("Target Gender Style:", ["masculine", "feminine"])

# -------------------------------------------------------------------
# Helper: Replicate AI API
# -------------------------------------------------------------------
def run_ai_transformation(pil_img, transform_type, age_val, gender_val):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        if transform_type == "Age Transformation":
            output = replicate.run(
                "yuval-alaluf/sam:9222a21c4421e0730221612ce5710f0e9f495b7cceed00a4cd8465105a372008",
                input={"image": buffer, "target_age": age_val}
            )
            return output
        elif transform_type == "Gender Swap":
            output = replicate.run(
                "easel/advanced-face-swap:latest",
                input={"swap_image": buffer, "mode": gender_val}
            )
            return output
    except Exception as e:
        st.error(f"Replicate API Error: {e}")
        return None

# -------------------------------------------------------------------
# Main Camera Input Pipeline
# -------------------------------------------------------------------
st.write("Take a snapshot to preview True View and AI transformations:")
camera_file = st.camera_input("Mirror Feed")

if camera_file is not None:
    # 1. Open image with PIL
    image = Image.open(camera_file)

    # 2. Perform Horizontal Flip via ImageOps (No OpenCV required!)
    if mode == "Standard Mirror (Flipped)":
        image = ImageOps.mirror(image)

    # 3. Render Mirror View
    st.image(image, caption=f"Viewport: {mode}", use_container_width=True)

    # 4. Process AI Transformation on clean original snapshot
    if ai_mode != "None":
        if st.button("🚀 Process AI Transformation"):
            with st.spinner("Synthesizing transformation..."):
                result_url = run_ai_transformation(Image.open(camera_file), ai_mode, target_age, gender_target)
                if result_url:
                    st.success("Transformation Complete!")
                    st.image(result_url, caption="AI Transformed Result", use_container_width=True)
