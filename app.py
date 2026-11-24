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
    sd = torch.load("best_model.pth", map_location=device)

    # fc ka size file se dekh lo
    out_f = sd['fc.weight'].shape[0]

    # Asli model = ResNet18, 1 channel
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, out_f)
    model.load_state_dict(sd)
    model.eval()
    return model, out_f

model, out_f = load_model()

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
        out = model(x)
        if out_f == 1:
            prob = torch.sigmoid(out.squeeze(1)).item()
        else:
            prob = torch.softmax(out, dim=1)[0][1].item()

    THR = 0.6457
    if prob >= THR:
        st.error(f"⚠️ PNEUMONIA - Confidence: {prob*100:.1f}%")
    else:
        st.success(f"✅ NORMAL - Confidence: {(1-prob)*100:.1f}%")
