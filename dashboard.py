import streamlit as st
from PIL import Image
import sys
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile
from src.predict import load_model, predict_and_save_with_box
from src.video_predict import process_video

# Environment diagnostics: show which Python the app is running under and torch availability
try:
    import torch
    _torch_info = f"torch {torch.__version__} (ok)"
except Exception as e:
    _torch_info = f"torch import failed: {e}"

_py_exe = sys.executable
st.set_page_config(page_title="Animal Prediction Dashboard", layout="wide")
st.title("🐾 Animal Prediction Dashboard")

# show environment in the sidebar so users can validate which Python Streamlit is using
with st.sidebar.expander("Environment"):
    st.write(f"Python executable: `{_py_exe}`")
    st.write(_torch_info)

# Load model
@st.cache_resource
def get_model():
    model, classes, img_size = load_model()
    return model, classes, img_size

model, classes, img_size = get_model()

# Sidebar: Image upload
def image_upload_section():
    st.sidebar.header("Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image
    return None

# Sidebar: Video upload
def video_upload_section():
    st.sidebar.header("Upload a Video")
    uploaded_vid = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"] , key='video')
    if uploaded_vid:
        # write temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
        tmp.write(uploaded_vid.read())
        tmp.flush()
        tmp.close()
        st.video(tmp.name)
        return tmp.name
    return None

# Prediction function
def predict_image(image):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    x = tfm(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_idx = int(torch.argmax(probs).item())
    pred_class = classes[top_idx] if classes else str(top_idx)
    pred_prob = float(probs[top_idx].item())
    return pred_class, pred_prob, probs.cpu().numpy()

# Main: Image upload and prediction
image = image_upload_section()
video_path = video_upload_section()
if video_path:
    st.markdown("### Video Uploaded")
    st.write(f"Uploaded file: `{os.path.basename(video_path)}`")
    # processing options in sidebar
    st.sidebar.subheader("Video processing options")
    scale = st.sidebar.slider("Output scale", 0.5, 2.0, 1.0, step=0.1, key='vid_scale')
    frame_step = st.sidebar.number_input("Frame step (process every N-th frame)", min_value=1, max_value=30, value=1, key='vid_frame_step')
    if st.button("Process video"):
        with st.spinner("Processing video — this can take some time depending on length and CPU/GPU..."):
            try:
                out_path, processed = process_video(video_path, out_root='outputs', scale=float(scale), frame_step=int(frame_step))
                st.success(f"Processed {processed} frames. Annotated video saved to: `{out_path}`")
                # Do NOT preview the annotated video inline — provide a direct download button only
                try:
                    with open(out_path, 'rb') as fh:
                        video_bytes = fh.read()
                    st.download_button("Download annotated video", data=video_bytes, file_name=os.path.basename(out_path), mime='video/mp4')
                except Exception:
                    st.info(f"Annotated video available at: {out_path}")
            except Exception as e:
                st.error(f"Video processing failed: {e}")
if image:
    # save annotated output and show path
    # write uploaded file to a temporary path so predict function can read it
    tmp_path = os.path.join('.','_tmp_upload.jpg')
    image.save(tmp_path)
    pred_class, pred_prob, out_path = predict_and_save_with_box(tmp_path, out_root='outputs', save=True)
    st.success(f"Prediction: {pred_class} ({pred_prob:.2%} confidence)")
    if out_path:
        st.write(f"Annotated image saved to: `{out_path}`")
        try:
            st.image(out_path, caption='Annotated output', use_column_width=True)
        except Exception:
            pass
    # show probability bar using model directly
    try:
        model, classes, img_size = get_model()
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        x = tfm(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        # build pandas Series so Streamlit labels bars nicely
        prob_series = pd.Series(probs, index=classes)
        st.subheader("Class probabilities")
        st.bar_chart(prob_series)
    except Exception:
        st.info("Could not compute probability graph (model/predict error).")

# Show sample images from each class
def show_samples():
    st.header("Sample Images by Class")
    data_dir = "data/animals/train"
    cols = st.columns(len(classes))
    for idx, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            img_path = os.path.join(class_dir, imgs[0])
            with cols[idx]:
                st.image(img_path, caption=cls, use_column_width=True)

show_samples()

# Optionally: Show model info
with st.expander("Model & Training Info"):
    st.write("**Model:** MobileNetV2")
    st.write(f"**Classes:** {', '.join(classes)}")
    st.write(f"**Input Size:** {img_size}x{img_size}")
    st.write("**Best model loaded from:** models/best_model.pt")
    st.write("**Author:** Your Team")

# Optionally: Show performance metrics if available
def show_metrics():
    metrics_path = "models/metrics.npy"
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True).item()
        st.header("Model Performance Metrics")
        st.write(metrics)
    else:
        st.info("No metrics file found. Retrain the model to generate metrics.")

show_metrics()
