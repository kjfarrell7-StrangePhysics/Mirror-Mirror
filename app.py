import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import mediapipe as mp
from mediapipe.solutions import face_mesh as mp_face_mesh
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

st.sidebar.header("🕸️ Face Landmark Tracking")
show_landmarks = st.sidebar.checkbox("Show Eye & Face Landmarks", value=False)

st.sidebar.header("✨ Generative AI Transformations")
ai_mode = st.sidebar.selectbox("AI Transformation:", ["None", "Age Transformation", "Gender Swap"])

target_age = 70
gender_target = "masculine"

if ai_mode == "Age Transformation":
    target_age = st.sidebar.slider("Target Age:", 10, 90, 70)
elif ai_mode == "Gender Swap":
    gender_target = st.sidebar.radio("Target Gender Style:", ["masculine", "feminine"])

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def draw_landmarks_pil(pil_img):
    """Draws facial points on PIL Image using MediaPipe."""
    img_np = np.array(pil_img)
    height, width, _ = img_np.shape
    
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_np)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw Iris & Eye Landmark Points
                for idx in range(468, 478):
                    pt = face_landmarks.landmark[idx]
                    x, y = int(pt.x * width), int(pt.y * height)
                    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="#00FFC8")
                    
    return draw_img

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
# Camera Viewport & Pipeline
# -------------------------------------------------------------------
st.write("Take a snapshot to preview True View and AI transformations:")
camera_file = st.camera_input("Mirror Feed")

if camera_file is not None:
    # 1. Load Image
    image = Image.open(camera_file)

    # 2. Handle Flip / True View
    if mode == "Standard Mirror (Flipped)":
        image = ImageOps.mirror(image)

    # 3. Optional Landmarks Overlay
    if show_landmarks:
        image = draw_landmarks_pil(image)

    # 4. Render Main Output
    st.image(image, caption=f"Viewport: {mode}", use_container_width=True)

    # 5. Process AI Transformations
    if ai_mode != "None":
        if st.button("🚀 Process AI Transformation"):
            with st.spinner("Synthesizing transformation..."):
                result_url = run_ai_transformation(Image.open(camera_file), ai_mode, target_age, gender_target)
                if result_url:
                    st.success("Transformation Complete!")
                    st.image(result_url, caption="AI Transformed Result", use_container_width=True)
