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
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        color: #2e7d32;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3299/3299935.png", width=100)
    st.title("Panduan AI")
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

st.markdown("<h1 class='stTitle'>♻️ EcoVision: Smart Waste</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #555; margin-bottom: 30px;'>
        Sistem kecerdasan buatan untuk membantu klasifikasi jenis sampah secara otomatis dan real-time.
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('model.pt') 

model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

with st.container():
    
    ctx = webrtc_streamer(
        key="waste-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "environment", 
            },
            "audio": False
        },
        async_processing=True,
    )

if ctx.state.playing:
    st.success("Kamera Aktif")
else:
    st.warning("Klik 'Start' untuk mengaktifkan kamera")

st.info("ℹ️ Izin akses kamera diperlukan untuk menjalankan deteksi objek.")