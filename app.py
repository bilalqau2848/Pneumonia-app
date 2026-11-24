import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# --- Page Setup ---
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩻")
st.title("🩻 Pneumonia Detection from Chest X-Ray")
st.write("Sirf Chest X-Ray upload karein. Cartoon ya doosri photo upload karne par app rok dega.")

# --- Model Load ---
@st.cache_resource
def load_my_model():
    model = load_model("pneumonia_model.h5")
    return model

model = load_my_model()

# --- GUARD FUNCTION (Cartoon Rokne Wala) ---
def is_xray_image(uploaded_image):
    img = np.array(uploaded_image.convert("RGB"))
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    # X-Ray me color nahi hota, cartoon me hota hai
    color_diff = np.mean(np.abs(r.astype(int) - g.astype(int)) + np.abs(g.astype(int) - b.astype(int)))
    if color_diff > 15:
        return False, "Ye X-Ray nahi lag rahi, is me color hai. Sirf Chest X-Ray upload karein."
    return True, "OK"

# --- Upload ---
uploaded_file = st.file_uploader("Chest X-Ray upload karein", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # PEHLE GUARD CHECK KAREGA
    is_valid, msg = is_xray_image(image)
    if not is_valid:
        st.error(f"⚠️ {msg}")
        st.stop()

    # AGAR X-RAY SAHI HAI TAB HI MODEL CHALEGA
    if st.button("Predict Karein"):
        # Image ko model ke size pe le aao (150x150)
        img_resized = image.convert("RGB").resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            st.error(f"Result: PNEUMONIA hai ({confidence*100:.1f}% confidence)")
        else:
            st.success(f"Result: NORMAL hai ({(1-confidence)*100:.1f}% confidence)")

        st.info("Note: Ye AI prediction hai, final decision doctor karega.")
