import cv2
import numpy as np
from live_inference import LiveInference
from ui_renderer import draw_ui

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

inference = LiveInference(hold_threshold=0.5)

print("[INFO] 실시간 수화 인식 시작. ESC로 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    live_label, current_state, sentence = inference.process_frame(frame)

    # UI 생성
    ui = draw_ui(frame, live_label, current_state, sentence)

    cv2.imshow("SilentHeart", ui)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
inference.close()