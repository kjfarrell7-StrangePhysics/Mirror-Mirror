import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps

st.set_page_config(page_title="AI Mirror", page_icon="🪞", layout="centered")
st.title("🪞 AI Mirror")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Sidebar Controls
st.sidebar.header("🎨 AR Alterations")
hair_color = st.sidebar.selectbox("Hair Color:", ["Natural", "Blue", "Green", "Red", "Black", "Grey"])
eye_color = st.sidebar.selectbox("Eye Color:", ["Natural", "Blue", "Green", "Red", "Brown"])
add_glasses = st.sidebar.checkbox("🕶️ Glasses", value=False)
add_mustache = st.sidebar.checkbox("👨 Mustache", value=False)
add_beard = st.sidebar.checkbox("🧔 Beard", value=False)
age_shift = st.sidebar.slider("👵 Age Shift:", 0, 100, 0)

COLOR_MAP = {
    "Blue": (255, 120, 30),
    "Green": (30, 200, 50),
    "Red": (50, 30, 230),
    "Black": (20, 20, 20),
    "Grey": (180, 180, 180),
    "Brown": (40, 75, 120)
}

st.write("### Take a Photo to Transform")
camera_file = st.camera_input("Mirror Camera")

if camera_file is not None:
    pil_img = Image.open(camera_file)
    pil_img = ImageOps.mirror(pil_img) # Mirror view
    
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = img_bgr.shape

            # Eye Recolor
            if eye_color != "Natural":
                left_iris = [468, 469, 470, 471, 472]
                right_iris = [473, 474, 475, 476, 477]
                mask = np.zeros((h, w), dtype=np.uint8)
                for iris_pts in [left_iris, right_iris]:
                    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_pts], np.int32)
                    cv2.fillConvexPoly(mask, pts, 255)
                overlay = img_bgr.copy()
                overlay[mask == 255] = COLOR_MAP[eye_color]
                img_bgr = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)

            # Facial Hair
            if add_mustache or add_beard:
                overlay = img_bgr.copy()
                if add_mustache:
                    stache_pts = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17]
                    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in stache_pts], np.int32)
                    cv2.fillPoly(overlay, [pts], (30, 30, 30))
                if add_beard:
                    jaw_pts = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
                    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in jaw_pts], np.int32)
                    cv2.fillPoly(overlay, [pts], (30, 30, 30))
                img_bgr = cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0)

            # Glasses
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

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Transformed Result", use_container_width=True)
