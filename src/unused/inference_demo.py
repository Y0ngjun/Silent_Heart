import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# 모델 불러오기
model_data = joblib.load("model/mlp_model.pkl")
model = model_data["model"]
label_encoder = model_data["label_encoder"]

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 상태 변수
prev_label = ""
state_start_time = None
current_state = ""
sentence = ""
HOLD_THRESHOLD = 1.0  # 상태 유지 시간

print("[INFO] 실시간 수화 예측 시작. ESC로 종료하세요.")

# 특수기호 매핑
symbol_map = {"QUESTION": "?", "EXCLAMATION": "!"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            if len(keypoints) == 42:
                wrist_x, wrist_y = keypoints[0], keypoints[1]

                x5, y5 = keypoints[10], keypoints[11]
                x17, y17 = keypoints[34], keypoints[35]
                dx = x5 - x17
                dy = y5 - y17
                scale = (dx**2 + dy**2) ** 0.5 + 1e-6

                relative_keypoints = []
                for i in range(0, len(keypoints), 2):
                    relative_x = (keypoints[i] - wrist_x) / scale
                    relative_y = (keypoints[i + 1] - wrist_y) / scale
                    relative_keypoints.extend([relative_x, relative_y])

                X_input = np.array(relative_keypoints).reshape(1, -1)
                pred = model.predict(X_input)
                label = label_encoder.inverse_transform(pred)[0]

                if label != prev_label:
                    state_start_time = time.time()
                    prev_label = label
                else:
                    if (
                        time.time() - state_start_time > HOLD_THRESHOLD
                        and label != "NOTSIGN"
                    ):
                        current_state = symbol_map.get(label, label)

                if label == "NOTSIGN" and current_state:
                    sentence += current_state + " "
                    current_state = ""
                    prev_label = ""
                    state_start_time = None

    else:
        label = "No hand"
        prev_label = ""
        state_start_time = None

    display_label = symbol_map.get(label, label)

    # 텍스트 출력
    cv2.putText(
        image,
        f"Live: {display_label}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        image,
        f"Current: {current_state}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        image,
        f"Sentence: {sentence.strip()}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Live Sign Recognition", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
