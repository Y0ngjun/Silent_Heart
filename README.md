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

**Silent Heart**는 Mediapipe와 MLP 분류기를 활용해 손 제스처를 인식하고 알파벳/기호로 번역하는 실시간 수어 인식 시스템입니다. 간편한 UI를 통해 누구나 손쉽게 사용 가능합니다.

## 🎥 데모

[![Silent Heart 데모](http://img.youtube.com/vi/RqqmJxP97tQ/0.jpg)](https://www.youtube.com/watch?v=RqqmJxP97tQ)  

    클릭 시 재생됩니다.

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
<p align="left"> <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv&logoColor=white" alt="OpenCV"/> <img src="https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google&logoColor=white" alt="MediaPipe"/> <img src="https://img.shields.io/badge/Numpy-1.x-blueviolet?logo=numpy&logoColor=white" alt="NumPy"/> <img src="https://img.shields.io/badge/Pandas-2.x-lightgrey?logo=pandas&logoColor=black" alt="Pandas"/> <img src="https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn"/> <img src="https://img.shields.io/badge/joblib-%3E1.3-green?logo=python&logoColor=white" alt="joblib"/> </p>

## 🚀 설치 및 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 손 제스처 데이터 수집
python src/data_collector.py

# 3. 데이터 전처리
python src/preprocessor.py

# 4. 모델 학습
python src/model_trainer.py

# 5. 실시간 수어 인식 실행
python src/main.py
```
## 🎮 사용법

1. `data_collector.py`에서 `LABEL = "A"` 등으로 수집할 알파벳 지정  
2. 수어 제스처를 카메라에 비추고 **스페이스바**로 저장  
3. 충분한 수의 데이터를 수집한 뒤 전처리 및 학습  
4. `main.py` 실행 후 웹캠 앞에서 수어하면 화면 오른쪽에 인식된 문자가 실시간 출력됨  
    - 짧은 시간 한가지 수어를 유지하고 NOTSIGN을 입력하면 수어가 출력됨

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
    Silent Heart is made with 💚 to give voice to your hands.
