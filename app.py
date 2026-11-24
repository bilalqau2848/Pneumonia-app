import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Chest X-Ray upload karein, model batayega NORMAL hai ya PNEUMONIA.")

MODEL_ID = "adham-shah/pediatric-chest-xray-pneumonia-vit"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Chest X-Ray Upload Karein", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    with st.spinner("Analyzing..."):
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        labels = model.config.id2label
        # labels: 0 = NORMAL, 1 = PNEUMONIA
        pred_id = int(probs.argmax().item())
        pred_label = labels[pred_id]
        conf = float(probs[pred_id]) * 100

        st.subheader(f"Result: {pred_label}")
        st.write(f"Confidence: {conf:.2f}%")
        st.progress(conf / 100)

        if pred_label.upper() == "PNEUMONIA":
            st.error("⚠️ Pneumonia Detected - Please consult a doctor.")
        else:
            st.success("✅ Normal - Pneumonia Not Detected.")

        st.write("**Probabilities:**")
        for i, label in labels.items():
            st.write(f"- {label}: {float(probs[i])*100:.2f}%")

st.caption("Educational demo only. Not for real medical diagnosis.")
