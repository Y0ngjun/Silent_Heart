import numpy as np
import mediapipe as mp
import joblib
import time
import cv2

class LiveInference:
    def __init__(self, model_path="model/mlp_model.pkl", hold_threshold=1.0):
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.label_encoder = model_data["label_encoder"]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
        )
        self.prev_label = ""
        self.state_start_time = None
        self.current_state = ""
        self.sentence = ""
        self.HOLD_THRESHOLD = hold_threshold
        self.symbol_map = {"QUESTION": "?", "EXCLAMATION": "!"}

    def process_frame(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        label = ""
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                if len(keypoints) == 21:
                    wrist_x, wrist_y = keypoints[0]
                    x5, y5 = keypoints[5]
                    x17, y17 = keypoints[17]
                    dx, dy = x5 - x17, y5 - y17
                    scale = (dx**2 + dy**2) ** 0.5 + 1e-6

                    relative_keypoints = []
                    for x, y in keypoints:
                        relative_keypoints.append((x - wrist_x) / scale)
                        relative_keypoints.append((y - wrist_y) / scale)

                    X_input = np.array(relative_keypoints).reshape(1, -1)
                    pred = self.model.predict(X_input)
                    label = self.label_encoder.inverse_transform(pred)[0]

                    if label != self.prev_label:
                        self.state_start_time = time.time()
                        self.prev_label = label
                    else:
                        if time.time() - self.state_start_time > self.HOLD_THRESHOLD and label != "NOTSIGN":
                            self.current_state = self.symbol_map.get(label, label)

                    if label == "NOTSIGN" and self.current_state:
                        self.sentence += self.current_state + " "
                        self.current_state = ""
                        self.prev_label = ""
                        self.state_start_time = None
        else:
            label = "No hand"
            self.prev_label = ""
            self.state_start_time = None

        display_label = self.symbol_map.get(label, label)
        return display_label, self.current_state, self.sentence.strip()

    def close(self):
        self.hands.close()
