![header](https://capsule-render.vercel.app/api?type=waving&color=0:2ecc71,100:1abc9c&height=250&section=header&text=Silent%20Heart&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=40)

# Silent Heart 💚

    조용한 마음을 전해보세요.
## 📌 목차
1. [소개](#-소개)
2. [데모](#-데모)
4. [기능 요약](#%EF%B8%8F-기능-요약)
5. [프로젝트 구조](#-프로젝트-구조)
6. [기술 스택](#-기술-스택)
7. [설치 및 실행 방법](#-설치-및-실행-방법)
8. [사용법](#-사용법)
9. [스크린샷 및 UI 설명](#%EF%B8%8F-스크린샷-및-ui-설명)
10. [커스터마이징 및 확장](#-커스터마이징-및-확장)
11. [트러블슈팅](#-트러블슈팅)
12. [기여 방법](#-기여-방법)
13. [라이선스](#-라이선스)

## 📖 소개

**Silent Heart**는 Python기반의 Mediapipe와 MLP 분류기를 활용해 손 제스처를 인식하고 알파벳/기호로 번역하는 실시간 수어 인식 시스템입니다. 간편한 UI를 통해 누구나 손쉽게 사용 가능합니다.

## 🎥 데모

### Hello World!
<a href="https://www.youtube.com/watch?v=RqqmJxP97tQ" target="_blank">
  <img src="http://img.youtube.com/vi/RqqmJxP97tQ/0.jpg" alt="Hello World! 데모">
</a>  

### A to Z
<a href="https://www.youtube.com/watch?v=LvayX4JgJXs" target="_blank">
  <img src="http://img.youtube.com/vi/LvayX4JgJXs/0.jpg" alt="A to Z 데모">
</a>

> 클릭 시 새 탭에서 재생됩니다.


## ⚙️ 기능 요약

- 🖐️ **21개 손 관절 좌표 추출**
- 🔠 **알파벳 + 기호 실시간 인식**
- 🚫 **NOTSIGN 제스처로 반복 방지 처리**
- 📹 **OpenCV 기반 실시간 UI 인터페이스**
- 🔧 **손목 기준 정규화 및 scaling 처리**
- 💬 **확정 라벨(state) + 문장(sentence) 동시 출력**

## 🗂 프로젝트 구조 
```
silent_heart/  
│  
├── data/  
│   ├── raw_landmarks/       # Original landmark CSV 데이터  
│   └── processed_data/      # 정규화된 학습용 CSV  
│  
├── model/  
│   └── mlp_model.pkl        # 학습된 MLP 모델 + 라벨 인코더  
│  
└── src/  
    ├── data_collector.py    # 키 입력으로 제스처 샘플 수집  
    ├── preprocessor.py      # 손목 기준 좌표로 전처리  
    ├── model_trainer.py     # MLP 학습 + 저장  
    ├── live_inference.py    # 실시간 예측 및 상태 유지  
    ├── ui_renderer.py       # UI 그리기 모듈  
    └── main.py              # 실행 메인 루프  
```
## 📚 기술 스택
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-4.11.0.86-brightgreen?logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10.21-orange?logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/Numpy-1.26.4-blueviolet?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-2.2.3-lightgrey?logo=pandas&logoColor=black" alt="Pandas"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.6.1-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/joblib-1.5.1-green?logo=python&logoColor=white" alt="joblib"/>
</p>

# 🚀 설치 및 실행 방법

## 🖥️ 실행 환경


- 운영체제: **Windows 11**
- 해상도: **1920 x 1080 (FHD)**
- 디스플레이 배율: **100%**

> ⚠️ 고해상도(예: 4K) 환경이나 배율이 125% 이상인 경우 UI가 비정상적으로 표시될 수 있습니다.

## 1. 프로젝트 다운로드


- GitHub Repository:  
  🔗 [https://github.com/Y0ngjun/Silent_Heart](https://github.com/Y0ngjun/Silent_Heart)

> ⚠️ **중요:** 프로젝트는 반드시 **영문 경로**에 저장해야 합니다.  
> 예를 들어, `C:\Users\YourName\Silent_Heart` 처럼 **한글이나 공백이 포함되지 않은 경로**를 사용하세요.

> Mediapipe 등 일부 라이브러리는 한글 경로에서 **리소스 파일을 불러오지 못해 실행이 실패**할 수 있습니다.

## 2. Python 3.10.10 설치


- 공식 다운로드 링크:  
  🔗 [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

> 설치 시 반드시 **"Add Python to PATH"** 옵션을 체크하세요.


## 3. 가상환경 생성 및 활성화

먼저 VSCode에서 클론한 프로젝트 폴더를 엽니다.

> **Tip:** VSCode에서 `파일 → 폴더 열기`를 통해 Silent_Heart 프로젝트 루트 디렉토리를 열어주세요.

**Git Bash 기준:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```


## 4. 필수 패키지 설치

> 설치 전, VSCode에서 Python 인터프리터를 프로젝트 내 가상환경(`.venv`)으로 설정해야 합니다.

### 🔧 VSCode에서 인터프리터 설정 방법:

1. VSCode 하단의 **Python 인터프리터 표시줄**을 클릭  
   (또는 `Ctrl + Shift + P` → "Python: 인터프리터 선택")
2. 목록에서 `.venv\Scripts\python.exe` 경로가 있는 항목을 선택

> 설정 후, VSCode 터미널을 다시 열어야 적용됩니다.

### ✅ 패키지 설치 (가상환경 활성화된 상태에서)

```bash
pip install -r requirements.txt
```


## 5. (선택) VSCode 디버그 설정 수정

`.vscode/launch.json` 파일에서 다음 항목을 수정합니다:

```json
"type": "debugpy"  →  "type": "python"
```


## 6. 프로젝트 실행

```bash
python src/main.py
```

# 📘 사용법

## 🔹 1. 수어 데이터 수집

새로운 수어 데이터를 수집하려면 아래 파일을 실행합니다:

```bash
python src/data_collector.py
```

### ✏️ 사용법
- 스크립트 내 `LABEL` 값을 변경하여 수집할 제스처 이름을 설정합니다.
- 실행 후 웹캠이 켜지고, **스페이스바를 누를 때마다 현재 손 제스처가 저장**됩니다.
- 수집된 데이터는 `data/raw_landmarks/{LABEL}.csv`로 저장됩니다.
- 원하는 의미의 커스텀 수어를 추가할 수 있습니다.


## 🔹 2. 데이터 전처리

수집된 데이터를 모델 학습에 적합한 형태로 변환하려면 다음 파일을 실행합니다:

```bash
python src/preprocessor.py
```

### ✏️ 사용법
- `data/raw_landmarks`에 있는 `.csv`들을 불러와 전처리한 뒤,
- `data/processed_data/final_dataset_full.csv` 형태로 저장합니다.
- 전처리 방식은 손 크기에 따른 스케일링이 적용되고 손목 기준 상대 좌표로 변환됩니다.


## 🔹 3. 모델 학습

전처리된 데이터를 기반으로 수어 분류 모델을 학습합니다:

```bash
python src/model_trainer.py
```

### ✏️ 사용법
- 모든 전처리된 `.csv`를 불러와 다중 클래스 분류 모델을 학습합니다.
- 학습된 모델은 `model/mlp_model.pkl`로 저장됩니다.
- 기본 모델은 **MLPClassifier (scikit-learn 기반)** 이며, 간단한 **GridSearchCV 기반 하이퍼파라미터 튜닝** 이 포함되어 있습니다.



## 🔹 4. 실시간 수어 인식 실행

실시간으로 웹캠을 통해 수어를 인식하려면 다음 파일을 실행합니다:

```bash
python src/main.py
```

### ✏️ 기능
- Mediapipe로 손 관절 인식 → 전처리 → 모델 예측 → 실시간 UI 출력
- 실시간 상태와 문장을 화면에 시각적으로 표시
- 알파벳, 구두점 등 지정된 수어를 실시간으로 인식

### ✏️ 사용법
- 특정 수어(알파벳/기호)를 **1초 이상 유지**하면 해당 동작이 **현재 state**로 확정됩니다.
- 이후 **"NOTSIGN" 수어를 입력하면 해당 state가 문장에 추가**됩니다.
- 여러 번 입력하려면 반드시 중간에 NOTSIGN을 포함해야 중복 출력되지 않습니다.


## ✅ 사용 전 필수 확인

- `model/mlp_model.pkl` 파일이 존재해야 `main.py`가 정상 작동합니다.
- 수어 라벨 간 구분을 위해 `"NOTSIGN"` 등의 구분 동작을 추가하면 정확도가 향상됩니다.


## 🎯 전체 실행 흐름 요약

1. `data_collector.py` – 라벨별 수어 수집  
2. `preprocessor.py` – 수집 데이터 전처리  
3. `model_trainer.py` – 모델 학습  
4. `main.py` – 실시간 인식 및 UI 실행


## 🖼️ 스크린샷 및 UI 설명
![image](https://github.com/user-attachments/assets/3d4dedae-b4a4-43b4-ac2a-586d9a4eebcc)

- **좌측:** 실시간 카메라 화면  
- **우측 상단:**
    - 현재 인식된 라벨 (주황색)  
    - 출력될 라벨 (초록색)  
- **우측 중앙:** 현재까지 출력된 문장  

## 🧩 커스터마이징 및 확장
- 🔡 새로운 알파벳 추가 → `data_collector.py`에서 라벨 변경 후 수집  
- 🆕 `"NOTSIGN"`, `"QUESTION"`, `"EXCLAMATION"` 등의 커스텀 기호 학습 가능  
- 🧠 모델 구조 변경 (예: TensorFlow, CNN 등)도 가능

## 🧯 트러블슈팅

| 문제                 | 해결법                                           |
|----------------------|--------------------------------------------------|
| 카메라 안 켜짐       | 다른 장치에서 카메라 점유 여부 확인             |
| 인식 정확도 낮음     | 데이터 수 부족 → 다양한 각도, 위치 수집         |
| 반복 문자 출력됨     | `hold_threshold` 조정 (`live_inference.py`)     |
| 모델 파일 오류       | `mlp_model.pkl` 경로 확인 또는 재학습            |

## 🤝 기여 방법

1. 이 저장소를 **fork**  
2. 새로운 브랜치 생성 후 수정  
3. **Pull Request** 작성

> 언제든 피드백과 아이디어를 환영합니다!


## 📄 라이선스

MIT License  
> 자유롭게 사용하고 개선해 주세요. 단, 출처는 남겨주시기 바랍니다.

---
     💚 Silent Heart는 손의 움직임에 목소리를 담기 위해 만들어졌습니다.  
     누구든지, 어디서든지 조용한 마음을 전할 수 있도록.