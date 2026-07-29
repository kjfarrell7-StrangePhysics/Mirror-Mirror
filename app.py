import streamlit as st
import replicate
import os
from PIL import Image

# -------------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Mirror", page_icon="🪞", layout="centered")
st.title("🪞 AI Generative Mirror")

# Retrieve Replicate API Token from Streamlit Secrets
if "REPLICATE_API_TOKEN" in st.secrets:
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("🎨 Transformation Settings")

# Hair Color
hair_color = st.sidebar.selectbox(
    "Hair Color:",
    ["Natural", "Vibrant Red", "Blue", "Green", "Black", "Grey", "Blonde", "Brown"]
)

# Eye Color
eye_color = st.sidebar.selectbox(
    "Eye Color:",
    ["Natural", "Blue", "Green", "Red", "Brown"]
)

# Facial Features
mustache_style = st.sidebar.checkbox("👨 Magnum P.I. Style Chevron Mustache", value=False)
beard_style = st.sidebar.checkbox("🧔 Full Beard", value=False)
add_glasses = st.sidebar.checkbox("🕶️ Glasses", value=False)

# Target Age
age_target = st.sidebar.slider("🎂 Target Age (Regression / Progression):", 5, 80, 45)

# -------------------------------------------------------------------
# Build Prompt Logic
# -------------------------------------------------------------------
prompt_elements = []

if hair_color != "Natural":
    prompt_elements.append(f"vibrant {hair_color.lower()} hair")

if eye_color != "Natural":
    prompt_elements.append(f"striking {eye_color.lower()} eyes")

if mustache_style:
    prompt_elements.append("a thick 1980s Magnum P.I. chevron mustache")

if beard_style:
    prompt_elements.append("a neatly groomed full beard")

if add_glasses:
    prompt_elements.append("wearing stylish eyeglasses")

prompt_elements.append(f"appearing as a {age_target} year old person")

transformation_prompt = ", ".join(prompt_elements)

# -------------------------------------------------------------------
# Main Feed & Pipeline
# -------------------------------------------------------------------
st.write("### Take a Photo")
camera_file = st.camera_input("Mirror Camera")

if camera_file is not None:
    st.image(camera_file, caption="Captured Snapshot", use_container_width=True)
    
    st.markdown("---")
    if st.button("✨ Apply Generative AI Transformations", type="primary", use_container_width=True):
        if "REPLICATE_API_TOKEN" not in os.environ:
            st.error("Missing Replicate API Token! Please add REPLICATE_API_TOKEN to your Streamlit App Secrets.")
        else:
            with st.spinner(f"AI is re-imagining your photo..."):
                try:
                    # Model identifier with explicit hash version
                    model_id = "stability-ai/stable-diffusion-img2img:15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d"
                    
                    output = replicate.run(
                        model_id,
                        input={
                            "image": camera_file,
                            "prompt": f"Photorealistic portrait of the same person, {transformation_prompt}, high resolution, natural lighting, highly detailed face",
                            "negative_prompt": "blurry, distorted face, extra limbs, bad proportions, low quality, cartoon, drawing",
                            "prompt_strength": 0.55  # Preserves facial structure while applying modifications
                        }
                    )
                    
                    # Display Result
                    if output:
                        st.success("Transformation Complete!")
                        st.image(output[0], caption=f"Transformed Result ({age_target} Y/O)", use_container_width=True)
                except Exception as e:
                    st.error(f"Error executing AI transformation: {e}")
