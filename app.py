import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import replicate
import io

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Mirror & Face Mesh App", page_icon="🪞", layout="centered")
st.title("🪞 AI Mirror with MediaPipe Mesh Tracking")

# Initialize MediaPipe Face Mesh & Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("🪞 Viewport Settings")
mode = st.sidebar.radio(
    "Mirror Mode:",
    ["Standard Mirror (Flipped)", "True View (How others see you)"]
)

st.sidebar.header("🕸️ MediaPipe Facial Mesh")
show_mesh = st.sidebar.checkbox("Draw Face Mesh Topology", value=True)
mesh_style = st.sidebar.selectbox(
    "Mesh Style:",
    ["Tesselation (Full Wireframe)", "Face Contours", "Irises & Eyes Only"]
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
# MediaPipe Mesh Processing Helper
# -------------------------------------------------------------------
def process_face_mesh(img_rgb, style_mode):
    """Processes frame with MediaPipe Face Mesh and renders requested landmark overlay."""
    annotated_img = img_rgb.copy()
    
    # Initialize FaceMesh task (refine_landmarks=True enables 478 points including irises)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if style_mode == "Tesselation (Full Wireframe)":
                    mp_drawing.draw_landmarks(
                        image=annotated_img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                elif style_mode == "Face Contours":
                    mp_drawing.draw_landmarks(
                        image=annotated_img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                elif style_mode == "Irises & Eyes Only":
                    # Render Irises if available
                    mp_drawing.draw_landmarks(
                        image=annotated_img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
    
    return annotated_img

def run_ai_transformation(pil_img, transform_type, age_val, gender_val):
    """Encodes snapshot and dispatches to Replicate API."""
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
# Main Camera Feed Pipeline
# -------------------------------------------------------------------
st.write("Take a snapshot to preview True View, MediaPipe mesh tracking, and AI transforms:")
camera_file = st.camera_input("Mirror Feed")

if camera_file is not None:
    # 1. Convert snapshot to numpy RGB
    image = Image.open(camera_file)
    img_np = np.array(image)

    # 2. Mirroring logic (Streamlit camera output defaults to raw un-mirrored mode)
    if mode == "Standard Mirror (Flipped)":
        img_np = cv2.flip(img_np, 1)

    # 3. Apply MediaPipe Face Mesh if enabled
    if show_mesh:
        img_np = process_face_mesh(img_np, mesh_style)

    final_pil = Image.fromarray(img_np)

    # 4. Render Main Output
    st.image(final_pil, caption=f"Viewport: {mode} | Filter: {mesh_style if show_mesh else 'Clean'}", use_container_width=True)

    # 5. Cloud AI Trigger
    if ai_mode != "None":
        if st.button("🚀 Process AI Transformation"):
            with st.spinner("Synthesizing transformation..."):
                result_url = run_ai_transformation(Image.open(camera_file), ai_mode, target_age, gender_target)
                if result_url:
                    st.success("Transformation Complete!")
                    st.image(result_url, caption="AI Transformed Result", use_container_width=True)
