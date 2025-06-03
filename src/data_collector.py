import cv2
import mediapipe as mp
import pandas as pd
import os

LABEL = "LABEL_NAME"  # <- 수집할 라벨, 실행 전 변경
SAVE_DIR = "data/raw_landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f"{LABEL}.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
data = []

print(f"[INFO] '{LABEL}' 수집 시작. 스페이스바를 누를 때마다 샘플이 저장됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])

            if cv2.waitKey(1) == 32:  # spacebar
                data.append(keypoints)
                print(f"[{LABEL}] 샘플 저장됨: {len(data)}개")

    else:
        text = "hand to center"
        font_scale = 1.0
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

    bottom_text = f"Label: {LABEL} | Press [Space] to save | [ESC] to quit"
    cv2.putText(image, bottom_text, (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Data Collection", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# 저장
cap.release()
cv2.destroyAllWindows()
hands.close()

if data:
    columns = [f"x{i}" if j % 2 == 0 else f"y{i}" for i in range(21) for j in range(2)]
    df = pd.DataFrame(data, columns=columns)
    df["label"] = LABEL
    df.to_csv(SAVE_PATH, index=False)
    print(f"[DONE] '{LABEL}' 데이터 {len(data)}개 저장 완료 → {SAVE_PATH}")
else:
    print("[WARNING] 저장된 데이터 없음.")
