import streamlit as st
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load Model
model = YOLO('best.pt')

st.title("Sistem Deteksi Jenis Sampah")
st.write("Arahkan kamera Anda ke objek : Organik, Anorganik, Kertas, atau Logam")

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # conf=0.5 berarti hanya menampilkan deteksi dengan tingkat keyakinan di atas 50%
        results = model.predict(img, conf=0.5)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    
webrtc_streamer(key="waste-detection", video_processor_factory=VideoProcessor)