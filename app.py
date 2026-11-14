"""Interactive Streamlit Web Application for Garbage Classification and Grad-CAM Explainability.

Enables drag-and-drop waste classification, confidence score visualization,
and real-time Grad-CAM visual attention mapping.
"""

import io
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from src.dataset import DEFAULT_CLASSES, load_classes
from src.explain import (
    create_gradcam_comparison,
    make_gradcam_heatmap,
    overlay_gradcam,
)
from src.model import build_model, load_model

# Category metadata with recycling tips and icons
CATEGORY_INFO: Dict[str, Dict[str, str]] = {
    "cardboard": {
        "icon": "📦",
        "bin": "Blue Recycling Bin",
        "tip": "Flatten boxes to save space. Keep dry and free of food residue.",
        "color": "#d97706",
    },
    "glass": {
        "icon": "🍾",
        "bin": "Glass / Recycling Bin",
        "tip": "Rinse containers before disposal. Remove metal/plastic lids.",
        "color": "#059669",
    },
    "metal": {
        "icon": "🥫",
        "bin": "Blue Recycling Bin",
        "tip": "Empty and rinse aluminum/steel cans. Highly recyclable.",
        "color": "#475569",
    },
    "paper": {
        "icon": "📰",
        "bin": "Blue Recycling Bin",
        "tip": "Recycle office paper, magazines, and envelopes. Avoid grease or oil.",
        "color": "#2563eb",
    },
    "plastic": {
        "icon": "🧴",
        "bin": "Blue Recycling Bin",
        "tip": "Check recycling number (#1 and #2 accepted everywhere). Rinse clean.",
        "color": "#7c3aed",
    },
    "trash": {
        "icon": "🗑️",
        "bin": "Black Landfill Bin",
        "tip": "Non-recyclable waste. Dispose in general landfill receptacle.",
        "color": "#dc2626",
    },
}


@st.cache_resource(show_spinner="Loading Neural Network Architecture...")
def get_model(model_path: str = "GarbageMLModel_CS549.keras", model_type: str = "custom_cnn"):
    """Loads a saved Keras model, downloads from Hugging Face if missing, or falls back to initialized baseline."""
    # 1. If custom path exists on disk, load it
    if model_path and os.path.isfile(model_path):
        try:
            model = load_model(model_path, compile_model=False)
            return model, f"Local Checkpoint ({os.path.basename(model_path)})"
        except Exception:
            pass

    # 2. If MobileNetV2 architecture selected
    if model_type == "mobilenet_v2":
        mb_candidates = ["GarbageMLModel_CS549_mobilenet_v2.keras", "mobilenet_v2.keras"]
        for cand in mb_candidates:
            if os.path.isfile(cand):
                try:
                    return load_model(cand, compile_model=False), f"Local Checkpoint ({cand})"
                except Exception:
                    pass
        # Build pre-trained ImageNet MobileNetV2
        mb_model = build_model(model_type="mobilenet_v2", compile_model=False)
        return mb_model, "MobileNetV2 (ImageNet Pre-trained Backbone)"

    # 3. Check local Custom CNN candidates
    candidates = ["GarbageMLModel_CS549.keras", "GarbageMLModel_CS549.h5", "model.h5"]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                model = load_model(cand, compile_model=False)
                return model, f"Local Checkpoint ({os.path.basename(cand)})"
            except Exception:
                continue

    # 4. Automatically download trained weights from Hugging Face repository
    try:
        from huggingface_hub import hf_hub_download

        hf_repo = "vincenthuynh/GarbageMLModel_CS549"
        downloaded_path = hf_hub_download(repo_id=hf_repo, filename="model.h5")
        model = load_model(downloaded_path, compile_model=False)
        return model, f"Hugging Face Hub ({hf_repo})"
    except Exception as e:
        fallback_model = build_model(model_type="custom_cnn", input_shape=(150, 150, 3), num_classes=6, compile_model=False)
        return fallback_model, f"Untrained Baseline (Error: {e})"


def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """Preprocesses a PIL Image into a normalized tensor."""
    img_rgb = img.convert("RGB").resize(target_size)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_prediction(model, tensor: np.ndarray, classes: List[str]) -> List[Dict[str, float]]:
    """Runs forward pass and returns sorted category probabilities."""
    preds = model.predict(tensor, verbose=0)[0]
    results = []
    for i, p in enumerate(preds):
        label = classes[i] if i < len(classes) else f"Class {i}"
        results.append({"class": label, "index": i, "confidence": float(p)})
    return sorted(results, key=lambda x: x["confidence"], reverse=True)


def main():
    st.set_page_config(
        page_title="Garbage Classification & Explainability",
        page_icon="🗑️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header Style
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .prediction-card {
            background-color: #f8fafc;
            border-left: 5px solid #2563eb;
            padding: 1.2rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🗑️ Garbage Classification & AI Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deep Learning CNN for automated waste sorting with real-time Grad-CAM visual attention mapping.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("⚙️ Settings & Controls")

    arch_options = ["Custom CNN (Baseline)", "MobileNetV2 (Transfer Learning)"]
    chosen_arch = st.sidebar.selectbox("Architecture Backbone", options=arch_options, index=0)
    selected_type = "mobilenet_v2" if "MobileNet" in chosen_arch else "custom_cnn"
    default_path = (
        "GarbageMLModel_CS549_mobilenet_v2.keras"
        if selected_type == "mobilenet_v2"
        else "GarbageMLModel_CS549.keras"
    )
    model_path = st.sidebar.text_input("Model Checkpoint", value=default_path)
    classes_path = st.sidebar.text_input("Classes Mapping", value="classes.json")
    classes = load_classes(classes_path) if os.path.isfile(classes_path) else list(DEFAULT_CLASSES)

    model, model_source = get_model(model_path=model_path, model_type=selected_type)

    if "Untrained" in model_source:
        st.sidebar.warning(f"⚠️ {model_source}")
    else:
        st.sidebar.success(f"✅ {model_source}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Grad-CAM Controls")

    # Detect conv layers dynamically from the loaded model
    conv_layers = [l.name for l in model.layers if "conv" in l.__class__.__name__.lower()]
    default_layer_idx = len(conv_layers) - 1 if conv_layers else 0
    target_layer = st.sidebar.selectbox(
        "Convolutional Layer",
        options=conv_layers if conv_layers else ["conv3"],
        index=default_layer_idx,
    )
    alpha = st.sidebar.slider("Heatmap Opacity (α)", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    colormap = st.sidebar.selectbox("Heatmap Colormap", options=["jet", "turbo", "viridis", "plasma", "magma"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 👨‍💻 Created by Vincent Huynh
        - [GitHub Repository](https://github.com/vintendohuynh/GarbageClassificationModel)
        - [Hugging Face Model](https://huggingface.co/vincenthuynh/GarbageMLModel_CS549)
        """
    )

    # Input Method Selection
    input_tab1, input_tab2 = st.tabs(["📂 Select Sample Image", "⬆️ Upload Your Own Image"])
    selected_image: Optional[Image.Image] = None
    image_name = "sample_image.jpg"

    with input_tab1:
        samples_dir = "assets/samples"
        available_samples = {}
        if os.path.isdir(samples_dir):
            for f in sorted(os.listdir(samples_dir)):
                if f.endswith((".jpg", ".png", ".jpeg")):
                    cat_name = os.path.splitext(f)[0]
                    available_samples[cat_name] = os.path.join(samples_dir, f)

        if available_samples:
            col_samples = st.columns(len(available_samples))
            sample_keys = list(available_samples.keys())
            chosen_sample = st.selectbox("Choose a pre-packaged waste sample:", options=sample_keys, index=0)
            sample_path = available_samples[chosen_sample]
            selected_image = Image.open(sample_path)
            image_name = os.path.basename(sample_path)
        else:
            st.info("No sample images found in assets/samples. Use the Upload tab to classify an image.")

    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload an image of garbage/waste (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name

    if selected_image is None:
        st.stop()

    st.markdown("---")

    # Run Prediction & Grad-CAM
    tensor = preprocess_image(selected_image)
    results = run_prediction(model, tensor, classes)
    top_pred = results[0]
    top_class = top_pred["class"]
    top_conf = top_pred["confidence"]
    info = CATEGORY_INFO.get(top_class, {"icon": "♻️", "bin": "Recycling", "tip": "Dispose responsibly.", "color": "#2563eb"})

    # Layout: Results & Breakdown
    col_img, col_metrics = st.columns([1, 1.3])

    with col_img:
        st.subheader("📸 Input Image")
        st.image(selected_image, caption=image_name, use_container_width=True)

    with col_metrics:
        st.subheader("🎯 Classification Result")
        st.markdown(
            f"""
            <div class="prediction-card" style="border-left-color: {info['color']};">
                <h2 style="margin: 0; color: #1e293b;">{info['icon']} {top_class.upper()}</h2>
                <p style="font-size: 1.25rem; font-weight: 600; color: {info['color']}; margin: 0.3rem 0;">
                    {top_conf * 100:.1f}% Confidence
                </p>
                <p style="margin: 0.3rem 0; color: #334155;"><strong>Recommended Disposal:</strong> {info['bin']}</p>
                <p style="margin: 0; color: #64748b; font-size: 0.95rem;"><em>{info['tip']}</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Category Probabilities")
        for res in results:
            cls = res["class"]
            conf = res["confidence"]
            icon = CATEGORY_INFO.get(cls, {}).get("icon", "•")
            col_label, col_bar = st.columns([1.2, 3])
            with col_label:
                st.write(f"{icon} **{cls.capitalize()}** ({conf * 100:.1f}%)")
            with col_bar:
                st.progress(min(1.0, max(0.0, conf)))

    # Grad-CAM Explainability Section
    st.markdown("---")
    st.subheader("🔍 Visual Attention Mapping (Grad-CAM)")
    st.write(
        "Grad-CAM (Gradient-weighted Class Activation Mapping) analyzes the gradients flowing into the final "
        f"convolutional layer (`{target_layer}`) to highlight the exact visual features driving the model's decision."
    )

    try:
        heatmap = make_gradcam_heatmap(
            img_array=tensor,
            model=model,
            last_conv_layer_name=target_layer,
            pred_index=top_pred["index"],
        )
        overlay = overlay_gradcam(selected_image, heatmap, alpha=alpha, colormap=colormap)

        # Side-by-side visualization
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Original Input**")
            st.image(selected_image, use_container_width=True)

        with c2:
            st.markdown(f"**2. Activation Heatmap ({colormap})**")
            # Convert 2D heatmap to colored preview
            cmap = plt.get_cmap(colormap)
            colored_hm = np.uint8(255 * cmap(heatmap)[:, :, :3])
            st.image(colored_hm, use_container_width=True)

        with c3:
            st.markdown("**3. Superimposed Attention Overlay**")
            st.image(overlay, use_container_width=True)

        # Download comparison option
        comparison_figure = create_gradcam_comparison(
            image_input=selected_image,
            heatmap=heatmap,
            superimposed_img=overlay,
            class_label=top_class,
            confidence=top_conf,
            colormap=colormap,
        )

        buf = io.BytesIO()
        comparison_figure.save(buf, format="PNG")
        st.download_button(
            label="📥 Download 3-Panel Grad-CAM Report",
            data=buf.getvalue(),
            file_name=f"gradcam_{top_class}_{os.path.splitext(image_name)[0]}.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"Grad-CAM computation failed on layer '{target_layer}': {e}")


if __name__ == "__main__":
    main()
