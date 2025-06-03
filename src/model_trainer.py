import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


def load_processed_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y


def train_model(X, y, model_output_path="model/mlp_model.pkl"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 하이퍼파라미터 탐색 공간 설정
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.005]
    }

    base_model = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    print("[INFO] 그리드 서치 시작...")
    grid.fit(X_train, y_train)
    print("[INFO] 그리드 서치 완료")
    print("[최적 파라미터]", grid.best_params_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"[검증 정확도] {acc:.4f}")
    print("[분류 리포트]")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({"model": best_model, "label_encoder": le}, model_output_path)
    print(f"[저장 완료] 모델 및 인코더 → {model_output_path}")


if __name__ == "__main__":
    data_path = "data/processed_data/final_dataset_full.csv"
    X, y = load_processed_data(data_path)
    train_model(X, y)
