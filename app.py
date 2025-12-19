import streamlit as st
from ultralytics import YOLO
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Smart Waste Detection", layout="centered")

st.title("Sistem Deteksi Sampah Real-Time")
st.write("Aplikasi ini mendeteksi 4 jenis sampah: Organik, Anorganik, Kertas, dan Logam.")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.predict(img, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
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

st.info("Catatan: Berikan izin akses kamera pada browser Anda. Gunakan pencahayaan yang cukup untuk hasil terbaik.")