import streamlit as st
import torch
import requests

st.title("DEBUG - Keys Check")
url = "https://huggingface.co/Bilalqau/Pneumonia-model/resolve/main/best_model.pth"
r = requests.get(url)
open("best_model.pth", "wb").write(r.content)
sd = torch.load("best_model.pth", map_location="cpu")

st.text(f"Total keys: {len(sd)}")
st.text("Pehli 30 keys:")

txt = ""
for k in list(sd.keys())[:30]:
    txt += f"{k} -> {tuple(sd[k].shape)}\n"
st.code(txt)

st.text("Aakhri 10 keys:")
txt2 = ""
for k in list(sd.keys())[-10:]:
    txt2 += f"{k} -> {tuple(sd[k].shape)}\n"
st.code(txt2)
