import pandas as pd
import numpy as np
import os


def preprocess_landmarks(df):
    # 손목 기준 상대 좌표
    wrist_x = df["x0"].values
    wrist_y = df["y0"].values

    # 손 크기 기준: x5 ~ x17 사이 거리 (엄지와 새끼손가락 루트 관절 사이)
    dx = df["x5"].values - df["x17"].values
    dy = df["y5"].values - df["y17"].values
    scale = np.sqrt(dx**2 + dy**2) + 1e-6  # 0으로 나누는 걸 방지

    processed_data = []
    for i in range(21):
        col_x = f"x{i}"
        col_y = f"y{i}"

        if col_x in df.columns and col_y in df.columns:
            relative_x = (df[col_x] - wrist_x) / scale
            relative_y = (df[col_y] - wrist_y) / scale
            processed_data.append(relative_x)
            processed_data.append(relative_y)

    processed_df = pd.DataFrame(np.array(processed_data).T)

    new_cols = []
    for i in range(21):
        new_cols.extend([f"rx{i}", f"ry{i}"])
    processed_df.columns = new_cols

    processed_df["label"] = df["label"]

    return processed_df


def load_and_preprocess_abcd_data(
    raw_data_dir="data/raw_landmarks",
    output_dir="data/processed_data",
    output_filename="final_dataset_abcd.csv",
    labels=["A", "B", "C", "D"],
):

    all_data_frames = []

    print(f"[{raw_data_dir}]에서 데이터 로드 시작...")

    for label in labels:
        file_path = os.path.join(raw_data_dir, f"{label}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data_frames.append(df)
            print(f"'{file_path}' 로드 완료. 샘플 수: {len(df)}")
        else:
            print(
                f"[경고] '{file_path}' 파일을 찾을 수 없습니다. 이 라벨의 데이터는 건너뜝니다."
            )

    if not all_data_frames:
        print(
            "[오류] 처리할 데이터가 없습니다. raw_landmarks 폴더에 CSV 파일이 있는지 확인하세요."
        )
        return None

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"모든 라벨의 데이터 통합 완료. 총 샘플 수: {len(combined_df)}")

    print("랜드마크 전처리(상대 좌표 변환) 시작...")
    processed_df = preprocess_landmarks(combined_df)
    print("랜드마크 전처리 완료.")

    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, output_filename)

    processed_df.to_csv(final_output_path, index=False)
    print(f"[완료] 전처리된 데이터 저장됨: {final_output_path}")

    return processed_df


if __name__ == "__main__":
    all_labels = [chr(i) for i in range(ord("A"), ord("Z") + 1)] + [
        "QUESTION",
        "EXCLAMATION",
        "NOTSIGN",
    ]
    processed_data = load_and_preprocess_abcd_data(
        output_filename="final_dataset_full.csv", labels=all_labels
    )

    if processed_data is not None:
        print("\n--- 전처리된 데이터 미리보기 (상위 5개 행) ---")
        print(processed_data.head())
        print(f"\n총 데이터 샘플 수: {len(processed_data)}")
        print(f"컬럼: {processed_data.columns.tolist()}")
    else:
        print("\n데이터 전처리 실패.")
