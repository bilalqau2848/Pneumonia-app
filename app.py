import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")
st.title("🩺 Pediatric Pneumonia Detector")
st.write("Chest X-Ray upload karo, model batayega Pneumonia hai ya Normal.")

# Model Load
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.efficientnet_b0(weights=None)
    # 1 channel fix - kyunki X-ray grayscale hai
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

uploaded = st.file_uploader("X-Ray ki pic yahan dalo", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    st.image(img, caption="Uploaded X-Ray", width=300)
    
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1)).item()
    
    THR = 0.6457
    if prob >= THR:
        st.error(f"⚠️ **PNEUMONIA** - Confidence: {prob*100:.1f}%")
        st.write("Doctor se zaroor check karwayen.")
    else:
        st.success(f"✅ **NORMAL** - Confidence: {(1-prob)*100:.1f}%")
        st.write("Pneumonia nahi lag raha.")
    
    st.progress(prob if prob>=THR else 1-prob)
