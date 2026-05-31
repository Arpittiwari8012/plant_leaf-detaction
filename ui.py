import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    layout="centered"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# --------------------------------
# IMPORTANT:
# CHECK THIS ORDER FROM:
# print(train_generator.class_indices)
# --------------------------------
class_names = [
    "healthy",
    "powdery",
    "rust",
    "not_leaf"
]

# --------------------------------
# IMAGE SIZE
# --------------------------------
IMG_SIZE = (224, 224)

# --------------------------------
# TITLE
# --------------------------------
st.title("🌿 Plant Disease Detector")

st.write("Upload a plant leaf image.")

# --------------------------------
# FILE UPLOADER
# --------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# PREDICTION FUNCTION
# --------------------------------
def predict_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMG_SIZE)

    # Convert to array
    img_array = img_to_array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)[0]

    # Highest prediction index
    predicted_index = np.argmax(predictions)

    # Confidence
    confidence = float(predictions[predicted_index])

    # Safe class selection
    if predicted_index < len(class_names):
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "unknown"

    return predicted_class, confidence, predictions

# --------------------------------
# MAIN
# --------------------------------
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict button
    if st.button("🔍 Predict"):

        result, confidence, predictions = predict_image(image)

        # --------------------------------
        # SHOW PROBABILITIES
        # --------------------------------
        st.subheader("📊 Prediction Probabilities")

        for i in range(len(predictions)):

            if i < len(class_names):
                label = class_names[i]
            else:
                label = f"extra_class_{i}"

            st.write(
                f"{label}: {predictions[i] * 100:.2f}%"
            )

        # --------------------------------
        # FINAL RESULT
        # --------------------------------
        st.subheader("🩺 Final Result")

        # Non leaf
        if result == "not_leaf":
            st.error("❌ NOT A LEAF")

        # Low confidence
        elif confidence < 0.60:
            st.warning("⚠️ Low Confidence Prediction")

        # Healthy
        elif result == "healthy":
            st.success(
                f"✅ HEALTHY LEAF\n\nConfidence: {confidence*100:.2f}%"
            )

        # Powdery
        elif result == "powdery":
            st.warning(
                f"⚠️ POWDERY DETECTED\n\nConfidence: {confidence*100:.2f}%"
            )

        # Rust
        elif result == "rust":
            st.warning(
                f"⚠️ RUST DETECTED\n\nConfidence: {confidence*100:.2f}%"
            )

        else:
            st.error("❌ Unknown Prediction")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow")