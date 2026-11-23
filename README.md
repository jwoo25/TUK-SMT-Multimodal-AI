# TUK-SMT-Multimodal-AI
SMT 공정 결함 진단을 위한 멀티모달 딥러닝(Vision + Sensor) 백본 최적화 프로젝트
# SMT 공정 결함 진단을 위한 멀티모달 딥러닝 백본 조합 최적화

> 한국공학대학교 인공지능학과 종합설계기획 프로젝트  
> **Team:** 정지우, 박은비

## 📌 프로젝트 개요
SMT(표면실장기술) 제조 공정의 **PCB 이미지(Vision)**와 **5초 시계열 설비 센서 데이터(온도·습도·진동·가속도·소음)**를 결합하여 미세 납땜 불량을 고정밀·실시간으로 진단하는 멀티모달 딥러닝 시스템입니다.

## 🎯 주요 연구 내용
- **고속 멀티모달 파이프라인:** AI-Hub SMT 전수 데이터(46,220쌍) 고유 ID 1:1 매핑 및 JSON 단일 파싱 구축
- **엄격한 데이터 분할 & 정규화:** 데이터 누수 0건 분할(Train 32,868 / Val 8,217 / 봉인 Test 5,135) 및 센서 `StandardScaler` 적용
- **도메인 특화 증강:** PCB 수직 촬영 대칭성 및 시계열 인과성을 보존하는 맞춤형 Augmentation 설계
- **단계적 백본 조합 최적화:** Vision(`ResNet50`, `EfficientNet-B0`, `ConvNeXt-Tiny`) × Sensor(`1D-CNN`, `Bi-LSTM`, `Transformer`) 통제 비교 실험
- **현장 적용성 검증:** 불량 미검출(FN) 최소화 튜닝, 경량화(ONNX/양자화) 및 Streamlit 진단 프로토타입 구현

## 📊 초도 실험 결과 (Baseline: Result 1)
| Model | Params | Latency (bs=1) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet50 + 1D-CNN (NoAug) | 24.13M | 8.92 ms | 99.08% | 99.67% | 98.93% | 99.30% | 0.9990 |

## 🛠️ Tech Stack
- **Language & Framework:** Python, PyTorch, Torchvision, Streamlit
- **Data & Analysis:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **Infrastructure:** Google Colab (T4 GPU), AWS GPU
