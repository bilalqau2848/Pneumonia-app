import streamlit as st
import torch
import requests

st.title("DEBUG - Keys Check")
url = "https://huggingface.co/Bilalqau/Pneumonia-model/resolve/main/best_model.pth"
r = requests.get(url)
open("best_model.pth", "wb").write(r.content)
sd = torch.load("best_model.pth", map_location="cpu")

st.write("Type:", type(sd))
if isinstance(sd, dict):
    keys = list(sd.keys())
    st.write("Total keys:", len(keys))
    st.write("Pehli 30 keys:")
    for k in keys[:30]:
        st.write(k, " -> ", tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k]))
    st.write("Aakhri 10 keys:")
    for k in keys[-10:]:
        st.write(k, " -> ", tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k]))
else:
    st.write("Ye dict nahi hai, pura model hai!")
    st.write(sd)
