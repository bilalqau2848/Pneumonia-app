import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")
st.title("🩺 Pediatric Pneumonia Detector")
st.write("Chest X-Ray upload karo, model batayega Pneumonia hai ya Normal.")

@st.cache_resource
def load_model():
    url = "https://huggingface.co/Bilalqau/Pneumonia-model/resolve/main/best_model.pth"
    r = requests.get(url)
    open("best_model.pth", "wb").write(r.content)
    device = torch.device("cpu")
    
    # Pehle weights dekh kar shape pata karo
    sd = torch.load("best_model.pth", map_location=device)
    
    # First conv ke channels check karo
    first_w = sd['features.0.0.weight']
    in_ch = first_w.shape[1]
    # Last layer ke outputs check karo
    last_w = sd['classifier.1.weight']
    out_f = last_w.shape[0]

    model = models.efficientnet_b0(weights=None)
    # Jo shape file me hai, wahi model me laga do
    model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_f)
    model.load_state_dict(sd)
    model.eval()
    return model, in_ch

model, in_ch = load_model()

# Agar model 1 channel ka hai to L, agar 3 ka hai to RGB
mode = "L" if in_ch == 1 else "RGB"
norm = [0.485] if in_ch == 1 else [0.485, 0.456, 0.406]
norm_std = [0.229] if in_ch == 1 else [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm, norm_std)
])

uploaded = st.file_uploader("X-Ray ki pic yahan dalo", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert(mode)
    st.image(img, caption="Uploaded X-Ray", width=300)
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1)).item()
    THR = 0.6457
    if prob >= THR:
        st.error(f"⚠️ PNEUMONIA - Confidence: {prob*100:.1f}%")
    else:
        st.success(f"✅ NORMAL - Confidence: {(1-prob)*100:.1f}%")
