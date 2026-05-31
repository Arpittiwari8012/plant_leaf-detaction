import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("model.h5")

# Same order as training output
class_names = ['healthy', 'powdery', 'rust']

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    score = np.max(prediction)
    class_idx = np.argmax(prediction)

    # ✅ Direct prediction (no "Not Leaf" condition)
    return class_names[class_idx], score