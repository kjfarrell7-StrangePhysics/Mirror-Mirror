import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
import random

# -------------------------------------------------------------------
# Page Config & Title
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Creative Mirror Arcade", page_icon="🪞", layout="centered")
st.title("🪞 AI Creative Mirror Arcade")
st.caption("100% Local & Free • Powered by OpenCV & MediaPipe")

# Initialize MediaPipe Solutions
mp_face_mesh = mp.solutions.face_mesh

# -------------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------------
st.sidebar.header("🕹️ Choose Arcade Mode")
app_mode = st.sidebar.radio(
    "Select Experience:",
    ["1. Stylization Engine", "2. Smart AR & Privacy Filters", "3. Interactive Physics Sandbox"]
)

# -------------------------------------------------------------------
# Mode 1: Stylization Engine Helper Functions
# -------------------------------------------------------------------
def apply_thermal(img_bgr, palette):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    colormap_type = cv2.COLORMAP_INFERNO if palette == "Inferno" else cv2.COLORMAP_JET
    return cv2.applyColorMap(gray, colormap_type)

def apply_sketch(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_ascii_matrix(img_bgr):
    # Create an 8-bit digital green matrix style filter
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Quantize levels
    gray = (gray // 32) * 32
    matrix_img = np.zeros_like(img_bgr)
    matrix_img[:, :, 1] = gray  # Put intensities purely into the Green channel
    return matrix_img

# -------------------------------------------------------------------
# Main Feed Input
# -------------------------------------------------------------------
st.write("### Snap a Photo to Run the FX Engine")
camera_file = st.camera_input("Arcade Camera")

if camera_file is not None:
    # Read image and mirror horizontally for natural feel
    pil_img = Image.open(camera_file)
    pil_img = ImageOps.mirror(pil_img)
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # ---------------------------------------------------------------
    # MODE 1: STYLIZATION ENGINE
    # ---------------------------------------------------------------
    if app_mode == "1. Stylization Engine":
        st.sidebar.subheader("🎨 Style Settings")
        style_choice = st.sidebar.selectbox(
            "Filter Style:", 
            ["Thermal / Heat Vision (Inferno)", "Thermal / Heat Vision (Jet)", "Pencil / Comic Sketch", "Retro Matrix Digital"]
        )
        
        if style_choice == "Thermal / Heat Vision (Inferno)":
            output_bgr = apply_thermal(img_bgr, "Inferno")
        elif style_choice == "Thermal / Heat Vision (Jet)":
            output_bgr = apply_thermal(img_bgr, "Jet")
        elif style_choice == "Pencil / Comic Sketch":
            output_bgr = apply_sketch(img_bgr)
        elif style_choice == "Retro Matrix Digital":
            output_bgr = apply_ascii_matrix(img_bgr)

    # ---------------------------------------------------------------
    # MODE 2: SMART AR & PRIVACY FILTERS
    # ---------------------------------------------------------------
    elif app_mode == "2. Smart AR & Privacy Filters":
        st.sidebar.subheader("🕶️ AR Settings")
        ar_choice = st.sidebar.selectbox(
            "AR Effect:",
            ["Privacy Anonymizer (Background Blur)", "Tracking Glasses & Mask"]
        )
        
        output_bgr = img_bgr.copy()
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            results = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                if ar_choice == "Privacy Anonymizer (Background Blur)":
                    # Build face bounding mask
                    face_mask = np.zeros((h, w), dtype=np.uint8)
                    hull_pts = np.array([
                        (int(p.x * w), int(p.y * h)) for p in landmarks
                    ], np.int32)
                    cv2.fillConvexPoly(face_mask, cv2.convexHull(hull_pts), 255)
                    
                    # Expand mask slightly
                    kernel = np.ones((25, 25), np.uint8)
                    face_mask = cv2.dilate(face_mask, kernel, iterations=2)
                    
                    # Heavy background blur
                    bg_blurred = cv2.GaussianBlur(img_bgr, (55, 55), 0)
                    
                    # Composite foreground face over blurred background
                    mask_3d = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR) / 255.0
                    output_bgr = (img_bgr * mask_3d + bg_blurred * (1 - mask_3d)).astype(np.uint8)
                    
                elif ar_choice == "Tracking Glasses & Mask":
                    # Eye coordinate anchors
                    p1 = (int(landmarks[33].x * w), int(landmarks[33].y * h))
                    p2 = (int(landmarks[263].x * w), int(landmarks[263].y * h))
                    radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) / 2.8)
                    
                    # Draw scaled cyber-visor/glasses
                    cv2.circle(output_bgr, p1, radius, (255, 200, 0), -1)
                    cv2.circle(output_bgr, p2, radius, (255, 200, 0), -1)
                    cv2.circle(output_bgr, p1, radius, (0, 0, 0), 3)
                    cv2.circle(output_bgr, p2, radius, (0, 0, 0), 3)
                    cv2.line(output_bgr, p1, p2, (0, 0, 0), 5)

    # ---------------------------------------------------------------
    # MODE 3: INTERACTIVE PHYSICS SANDBOX
    # ---------------------------------------------------------------
    elif app_mode == "3. Interactive Physics Sandbox":
        st.sidebar.subheader("⚽ Physics Settings")
        particle_type = st.sidebar.selectbox("Particle Type:", ["Bouncy Balls", "Confetti Rain"])
        num_particles = st.sidebar.slider("Particle Density:", 20, 150, 60)
        
        output_bgr = img_bgr.copy()
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            results = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            # Head barrier y-level
            head_y = h // 2
            if results.multi_face_landmarks:
                pts = results.multi_face_landmarks[0].landmark
                # Forehead landmark index 10
                head_y = int(pts[10].y * h)
            
            # Simulate particle collision physics resting on top of head boundary
            random.seed(42)  # Deterministic seed for snapshot rendering
            for _ in range(num_particles):
                px = random.randint(20, w - 20)
                py = random.randint(10, max(head_y - 5, 15))  # Keep particles above head line
                
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(150, 255))
                
                if particle_type == "Bouncy Balls":
                    size = random.randint(8, 18)
                    cv2.circle(output_bgr, (px, py), size, color, -1)
                    cv2.circle(output_bgr, (px, py), size, (0, 0, 0), 2)
                else:
                    # Confetti squares
                    sz = random.randint(6, 14)
                    cv2.rectangle(output_bgr, (px, py), (px + sz, py + sz), color, -1)

    # Render output
    st.image(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB), caption="Arcade Render", use_container_width=True)
