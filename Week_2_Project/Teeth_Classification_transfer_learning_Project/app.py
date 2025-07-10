import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Constants
MODEL_PATH = "model_checkpoint/efficientnetb5_v1_best.h5"
IMAGE_SIZE = (256, 256)
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
CLASS_DESCRIPTIONS = {
    "CaS": "Calculus (Tartar)",
    "CoS": "Caries (Tooth Decay)",
    "Gum": "Gum Disease",
    "MC": "Mouth Cancer",
    "OC": "Oral Cancer",
    "OLP": "Oral Lichen Planus",
    "OT": "Oral Thrush",
}


def load_model():
    """Load the pre-trained EfficientNetB5 model."""
    # Check if model file exists
    import os

    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file not found at: {MODEL_PATH}")
        st.error("Please ensure the model file exists at the specified path.")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.error("Please ensure the model file exists at the specified path.")
        return None


def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image for prediction using TensorFlow.
    Reads the image file, decodes, resizes, normalizes, and adds batch dimension.
    """
    try:
        file_bytes = uploaded_file.read()
        # Reset file pointer for potential future reads
        uploaded_file.seek(0)
        img_tensor = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, IMAGE_SIZE)
        img_tensor = img_tensor / 255.0
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        return img_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict(uploaded_file, model):
    """Make prediction on the uploaded image file."""
    try:
        processed_image = preprocess_image(uploaded_file)
        if processed_image is None:
            return None, None, None

        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        predicted_class = CLASS_NAMES[predicted_class_idx]
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None


def main():
    # Page configuration
    st.set_page_config(
        page_title="Dental Classification AI",
        page_icon="🦷",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Header section
    st.markdown("# 🦷 Dental Condition Classification")
    st.markdown("### AI-Powered Dental Image Analysis")

    # Information section
    with st.expander("ℹ️ How to use this app", expanded=False):
        st.markdown(
            """
        **Instructions:**
        1. Upload a clear dental image (JPG, JPEG, or PNG format)
        2. Wait for the AI to analyze the image
        3. Review the classification results and confidence scores

        **What this app can detect:**
        """
        )
        # Use CLASS_DESCRIPTIONS for better information display
        for class_code, description in CLASS_DESCRIPTIONS.items():
            st.markdown(f"- **{class_code}**: {description}")

    st.markdown("---")

    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model = load_model()

    if model is None:
        st.error("❌ Failed to load the model. Please check the model file path.")
        st.stop()
    else:
        st.success("✅ AI model loaded successfully!")

    st.markdown("---")

    # File uploader section
    st.markdown("### 📤 Upload Your Dental Image")
    uploaded_file = st.file_uploader(
        "Choose a dental image file...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Maximum file size: 200MB",
    )

    if uploaded_file is not None:
        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display uploaded image
            st.markdown("#### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image, caption=f"File: {uploaded_file.name}", use_container_width=True
            )

        with col2:
            # Make prediction
            st.markdown("#### 🔍 Analysis Results")
            with st.spinner("🔄 Analyzing image..."):
                predicted_class, confidence, all_predictions = predict(
                    uploaded_file, model
                )

            if predicted_class is not None:
                # Display main prediction
                st.success(f"**Predicted Condition**: {predicted_class}")
                st.info(f"**Description**: {CLASS_DESCRIPTIONS[predicted_class]}")
                st.metric("Confidence", f"{confidence:.1f}%")

        # Display detailed results below both columns
        if predicted_class is not None:
            st.markdown("---")
            st.markdown("### 📊 Detailed Confidence Scores")

            # Create a more visual representation of confidence scores
            for class_name, prob in zip(CLASS_NAMES, all_predictions):
                confidence_pct = float(prob) * 100
                st.write(
                    f"**{class_name}** ({CLASS_DESCRIPTIONS[class_name]}): {confidence_pct:.2f}%"
                )
                st.progress(confidence_pct / 100)


if __name__ == "__main__":
    main()
