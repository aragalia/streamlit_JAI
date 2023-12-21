import streamlit as st
import numpy as np
import cv2
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av



st.title("Deteksi Kualitas Kematangan pada buah Jeruk")
st.markdown('silahkan klik "START" untuk memulai webcam dan klik "STOP" untuk mengakhiri webcam')

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

  

class_label = {
    0: "Matang",
    1: "Belum Matang"
}

def draw_bounding_boxes(pred_tensor, result):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    size_of_tensor = list(pred_tensor.size())
    rows = size_of_tensor[0]
    for i in range(0, rows):
        cv2.rectangle(result, (int(pred_tensor[i,0].item()), int(pred_tensor[i,1].item())), 
                      (int(pred_tensor[i,2].item()), int(pred_tensor[i,3].item())), (0, 0, 255), 2)

        text = class_label[int(pred_tensor[i,5].item())] +" " + str(round(pred_tensor[i,4].item(), 2))

        image = cv2.putText(result, text, (int(pred_tensor[i,0].item())+5, int(pred_tensor[i,1].item())), 
                            font, fontScale, (0, 0, 255), 2)
        
    return result

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        prediction = model(img)
        print(prediction)
        result_img = draw_bounding_boxes(prediction.xyxy[0], img)
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
