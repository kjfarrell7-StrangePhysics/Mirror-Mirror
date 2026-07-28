import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageOps

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Live Mirror", page_icon="🪞", layout="centered")
st.title("🪞 AI Live Mirror")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("🪞 Viewport Settings")
mode = st.sidebar.radio(
    "Mirror View:",
    ["Standard Mirror (Flipped)", "True View (How others see you)"]
)

st.sidebar.header("🎨 Live AR Alterations")

# Hair Color
hair_color = st.sidebar.selectbox(
    "Hair Color:",
    ["Natural", "Blue", "Green", "Red", "Black", "Grey"]
)

# Eye Color
eye_color = st.sidebar.selectbox(
    "Eye Color:",
    ["Natural", "Blue", "Green", "Red", "Brown"]
)

# Facial Accessories & Features
add_glasses = st.sidebar.checkbox("🕶️ Glasses", value=False)
add_mustache = st.sidebar.checkbox("👨 Mustache", value=False)
add_beard = st.sidebar.checkbox("🧔 Beard", value=False)

# Age Filter
age_shift = st.sidebar.slider("👵 Age Shift (Wrinkle & Tone Filter):", 0, 100, 0)

# Action Button
st.sidebar.markdown("---")
apply_button = st.sidebar.button("⚡ Apply AI Transformations", type="primary", use_container_width=True)

# -------------------------------------------------------------------
# Helper Overlay Functions
# -------------------------------------------------------------------
COLOR_MAP = {
    "Blue": (255, 120, 30),
    "Green": (30, 200, 50),
    "Red": (50, 30, 230),
    "Black": (20, 20, 20),
    "Grey": (180, 180, 180),
    "Brown": (40, 75, 120)
}

def apply_eye_recolor(img_bgr, landmarks, color_name):
    if color_name == "Natural":
        return img_bgr
    
    h, w, _ = img_bgr.shape
    # Left and Right Iris Indices in MediaPipe Refined Mesh
    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    for iris_pts in [left_iris, right_iris]:
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_pts], np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        
    color_bgr = COLOR_MAP[color_name]
    overlay = img_bgr.copy()
    overlay[mask == 255] = color_bgr
    return cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)

def apply_facial_hair(img_bgr, landmarks, mustache=False, beard=False):
    h, w, _ = img_bgr.shape
    overlay = img_bgr.copy()
    
    if mustache:
        # Upper Lip Outer Contour
        stache_pts = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17]
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in stache_pts], np.int32)
        cv2.fillPoly(overlay, [pts], (30, 30, 30))
        
    if beard:
        # Jawline Contour
        jaw_pts = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in jaw_pts], np.int32)
        cv2.fillPoly(overlay, [pts], (30, 30, 30))
        
    return cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)

def apply_glasses_overlay(img_bgr, landmarks):
    h, w, _ = img_bgr.shape
    # Left eye center ~33, Right eye center ~263
    p1 = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    p2 = (int(landmarks[263].x * w), int(landmarks[263].y * h))
    
    radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) / 3)
    
    cv2.circle(img_bgr, p1, radius, (20, 20, 20), 4)
    cv2.circle(img_bgr, p2, radius, (20, 20, 20), 4)
    cv2.line(img_bgr, (p1[0] + radius, p1[1]), (p2[0] - radius, p2[1]), (20, 20, 20), 4)
    return img_bgr

def apply_age_filter(img_bgr, intensity):
    if intensity == 0:
        return img_bgr
    # High-pass filter emulation for skin texture contrast (wrinkle simulation)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    high_pass = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    high_pass_bgr = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
    
    alpha = (intensity / 100.0) * 0.5
    return cv2.addWeighted(high_pass_bgr, alpha, img_bgr, 1 - alpha, 0)

# -------------------------------------------------------------------
# Video Feed & Pipeline (Session State Enabled)
# -------------------------------------------------------------------
st.write("### Live Camera Feed")

# Initialize session state for captured image
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None

camera_file = st.camera_input("Mirror Feed")

# Store photo in session state when taken
if camera_file is not None:
    pil_img = Image.open(camera_file)
    if mode == "Standard Mirror (Flipped)":
        pil_img = ImageOps.mirror(pil_img)
    st.session_state.captured_img = np.array(pil_img)

# Process photo if present in session state
if st.session_state.captured_img is not None:
    img_np = st.session_state.captured_img.copy()
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Execute transformations when button is clicked
    if apply_button:
        with st.spinner("Applying AI alterations..."):
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Apply Selected Transformations
                    img_bgr = apply_eye_recolor(img_bgr, landmarks, eye_color)
                    
                    if add_mustache or add_beard:
                        img_bgr = apply_facial_hair(img_bgr, landmarks, mustache=add_mustache, beard=add_beard)
                        
                    if add_glasses:
                        img_bgr = apply_glasses_overlay(img_bgr, landmarks)
                        
                    img_bgr = apply_age_filter(img_bgr, age_shift)
                else:
                    st.warning("No face detected in frame. Try adjusting lighting or moving closer.")

    # Render Output Frame
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Transformed Output", use_container_width=True)
