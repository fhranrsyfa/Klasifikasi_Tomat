import streamlit as st
import cv2
import numpy as np
import joblib
import os
import shutil
from skimage.filters import gabor
from rembg import remove 
from PIL import Image
import io

setup_assets()

st.set_page_config(
    page_title="Tomato AI - Kesegaran",
    page_icon="🍅",
    layout="centered"
)

def setup_assets():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://raw.githubusercontent.com/fhranrsyfa/Klasifikasi_Tomat/main/bg_tomat.jpg");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



@st.cache_resource
def load_assets():
    model_name = 'model_tomat.pkl'
    scaler_name = 'scaler.pkl'
    
    try:
        model = joblib.load(os.path.join(os.path.dirname(__file__), model_name))
        scaler = joblib.load(os.path.join(os.path.dirname(__file__), scaler_name))
        return model, scaler
    except Exception as e:
        st.error(f"Error detail: {e}")
        return None, None
model, scaler = load_assets()

def extract_features_with_stats(image_gray):
    orientasi_derajat = [0, 45, 90, 135]
    thetas = [np.deg2rad(d) for d in orientasi_derajat]
    frequencies = [0.1, 0.2]
    features = []
    
    img_norm = image_gray / 255.0
    for f in frequencies:
        for t_rad in thetas:
            filt_real, _ = gabor(img_norm, frequency=f, theta=t_rad)
            features.append(np.mean(filt_real))
            features.append(np.var(filt_real))
    
    avg_mean = np.mean(features[0::2])
    avg_var = np.mean(features[1::2])
    return features, avg_mean, avg_var

st.markdown("""
    <div class="main-navbar"><div class="nav-content">
    <span class="nav-logo">🍅</span><span class="nav-title">FreshnessOfTomatoes</span>
    </div></div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>Tomato Freshness Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Analisis Tekstur Gabor Filter & Random Forest.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("""
    <div style='text-align: center; margin-top: -190px; pointer-events: none; padding-bottom: 110px;'>
        <div style='font-size: 55px; color: #4bcba3; margin-bottom: 5px;'>☁️</div>
        <div style='color: #555; font-size: 16px; font-weight: 500;'>
            Drag & drop your tomato image here<br>
            <span style='font-size: 13px; color: #888;'>or click to browse</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    with st.spinner('Menghapus background...'):
        no_bg = remove(input_image)
        
        white_bg = Image.new("RGB", no_bg.size, (255, 255, 255))
        
        white_bg.paste(no_bg, mask=no_bg.split()[3])
        
        img_bgr = cv2.cvtColor(np.array(white_bg), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    


    c1, mid, c2 = st.columns([1, 2, 1])
    with mid:
        st.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Preview (Background Removed)", use_container_width=True)
        detect_clicked = st.button("Classify Tomato", use_container_width=True)

    if detect_clicked:
        if model and scaler:
            with st.spinner('Menganalisis parameter tekstur...'):
                fitur, val_mean, val_var = extract_features_with_stats(img_gray)
                fitur_final = scaler.transform([fitur])
                prediksi_raw = model.predict(fitur_final)[0]
                label = str(prediksi_raw).lower()

                if "segar" in label and "tidak" not in label:
                    res_title, res_icon, res_color = "Segar", "✔", "#28a745"
                    detail = "Kulit terdeteksi halus dan kencang. Kondisi prima."
                elif "tidak_segar" in label:
                    res_title, res_icon, res_color = "Tidak Segar", "⚠", "#ffc107"
                    detail = "Terdeteksi pola kerutan halus pada tekstur kulit."
                else:
                    res_title, res_icon, res_color = "Busuk", "✖", "#8b0000"
                    detail = "Terdeteksi kerusakan jaringan kulit atau pembusukan."

                st.markdown(f"""
                <div class="result-container" style="background-color: #fce8e6; border-left: 8px solid #3a1e1e; padding: 25px; border-radius: 15px; margin-top: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <span style="color: {res_color}; border: 2px solid {res_color}; border-radius: 50%; width: 35px; height: 35px; display: inline-flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: bold;">{res_icon}</span>
                        <span style="color: #3a1e1e; font-size: 24px; font-weight: 800;">{res_title}</span>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #eee;">
                        <p style="color: #3a1e1e; font-size: 16px; margin-bottom: 15px;"><b>Analisis:</b> {detail}</p>
                        <div style="border-top: 1px solid #f0f0f0; padding-top: 10px;">
                            <table style="width: 100%; color: #555; font-size: 13px; font-family: monospace;">
                                <tr><td>Mean Gabor</td><td style="text-align: right;">{val_mean:.6f}</td></tr>
                                <tr><td>Variance Gabor</td><td style="text-align: right;">{val_var:.6f}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Model/Scaler tidak ditemukan!")

st.markdown("""
    <div class="footer-container">
        <div class="footer-content">
            <span style="font-size: 20px;">🍅</span> <b>FreshnessOfTomatoes</b><br>
            <small>© 2026 Skripsi Project - Gabor & Random Forest</small>
        </div>
    </div>
""", unsafe_allow_html=True)
