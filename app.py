import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Mirror", page_icon="🪞", layout="centered")
st.title("🪞 AI Mirror")

# Initialize MediaPipe Solutions
mp_face_mesh = mp.solutions.face_mesh

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("🎨 AR Alterations")

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
age_shift = st.sidebar.slider("👵 Age Shift:", 0, 100, 0)

# Color lookup dictionary (BGR format)
COLOR_MAP = {
    "Blue": (255, 120, 30),
    "Green": (30, 200, 50),
    "Red": (50, 30, 230),
    "Black": (20, 20, 20),
    "Grey": (180, 180, 180),
    "Brown": (40, 75, 120)
}

# -------------------------------------------------------------------
# Camera & Transformations
# -------------------------------------------------------------------
st.write("### Snap a Photo to Apply Alterations")
camera_file = st.camera_input("Mirror Camera")

if camera_file is not None:
    # Load and mirror the input image
    pil_img = Image.open(camera_file)
    pil_img = ImageOps.mirror(pil_img)
    
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # 1. Hair Recoloring (Top region heuristic based on head orientation)
    if hair_color != "Natural":
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                pts = res.multi_face_landmarks[0].landmark
                # Top forehead boundary ~10
                top_y = int(pts[10].y * h)
                hair_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(hair_mask, (0, 0), (w, max(top_y, 10)), 255, -1)
                
                overlay = img_bgr.copy()
                overlay[hair_mask == 255] = COLOR_MAP[hair_color]
                img_bgr = cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0)

    # 2. Facial Landmark Mesh Transformations
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eye Recolor
            if eye_color != "Natural":
                left_iris = [468, 469, 470, 471, 472]
                right_iris = [473, 474, 475, 476, 477]
                iris_mask = np.zeros((h, w), dtype=np.uint8)
                for iris_pts in [left_iris, right_iris]:
                    pts_array = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_pts], np.int32)
                    cv2.fillConvexPoly(iris_mask, pts_array, 255)
                overlay = img_bgr.copy()
                overlay[iris_mask == 255] = COLOR_MAP[eye_color]
                img_bgr = cv2.addWeighted(overlay, 0.65, img_bgr, 0.35, 0)

            # Facial Hair
            if add_mustache or add_beard:
                overlay = img_bgr.copy()
                if add_mustache:
                    stache_pts = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17]
                    pts_array = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in stache_pts], np.int32)
                    cv2.fillPoly(overlay, [pts_array], (30, 30, 30))
                if add_beard:
                    jaw_pts = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
                    pts_array = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in jaw_pts], np.int32)
                    cv2.fillPoly(overlay, [pts_array], (30, 30, 30))
                img_bgr = cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)

            # Glasses Overlay
            if add_glasses:
                p1 = (int(landmarks[33].x * w), int(landmarks[33].y * h))
                p2 = (int(landmarks[263].x * w), int(landmarks[263].y * h))
                radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) / 3)
                cv2.circle(img_bgr, p1, radius, (20, 20, 20), 4)
                cv2.circle(img_bgr, p2, radius, (20, 20, 20), 4)
                cv2.line(img_bgr, (p1[0] + radius, p1[1]), (p2[0] - radius, p2[1]), (20, 20, 20), 4)

            # Age Filter
            if age_shift > 0:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (0, 0), 3)
                high_pass = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
                high_pass_bgr = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
                alpha = (age_shift / 100.0) * 0.5
                img_bgr = cv2.addWeighted(high_pass_bgr, alpha, img_bgr, 1 - alpha, 0)

    # Display Final Output Image
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Transformed Result", use_container_width=True)
