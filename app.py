import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="EcoVision - Smart Waste Detector",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
    <style>
    .stTitle {
        color: #2e7d32;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.title("♻️ Panduan EcoVision")
    st.markdown("""
    ### 🏷️ Kelas Deteksi:
    - 🍃 **Organik**
    - 🧴 **Anorganik**
    - 📝 **Kertas**
    - 🥫 **Logam**

    ### 💡 Tips:
    1. Pastikan cahaya terang.
    2. Objek tidak terlalu jauh.
    """)
    
st.markdown("<h1 class='stTitle'>EcoVision: Smart Waste</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('model.pt') 

model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

ctx = webrtc_streamer(
    key="waste-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "facingMode": "environment", 
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("✅ Sistem Berjalan")
else:
    st.info("💡 Tekan **START** untuk memuat kamera.")