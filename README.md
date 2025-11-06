## 교대근무표 OCR 인식 모델 
이 프로젝트는 교대근무표 이미지를 자동으로 인식하고, 각 셀 단위의 근무타입(Day, Night, Evening, Off)을 분류하여 JSON 형식으로 추출하는 OCR 기반 AI 모델입니다.
딥러닝 분류기(tflite)와 OpenCV 기반 전처리 파이프라인을 결합하여 다양한 근무표 형태를 정규화하고 자동 인식하도록 설계되었습니다. <br>
<br>
이미지 전처리부터 CNN 학습, TFLite 변환까지 전 과정을 하나의 Jupyter Notebook 파일(**3class_final_final(2).ipynb**)에서 실행할 수 있습니다.

### 프로젝트 개요
- **목표**<br>
교대근무표(엑셀, PDF, 사진 등)를 자동 인식하여 근무 타입(D/N/E/OFF) 및 이름, 날짜 정보를 추출<br>
- **핵심기능**<br>
1. 근무표 이미지 전처리 (표 구조 검출, 투시 보정)
2. 셀 단위 분할 및 OCR 인식
3. CNN 기반 근무 타입 분류 (Day/Night/Evening/None)
4. 결과를 JSON 형태로 자동 변환

### 모델 구조
모델은 Keras Sequential 구조로 구성되며, 입력된 셀 이미지를 32×32 크기로 정규화한 후 CNN을 통해 D/N/E 클래스를 분류합니다.<br>
```
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])
```
입력: 흑백 셀 이미지 (32×32)
출력: 3개 클래스 (D, N, E)
활성화함수: ReLU, Softmax
최적화: Adam, 학습률 0.001
손실 함수: categorical_crossentropy

### 학습 데이터셋
데이터 출처: [Kaggle English Font Dataset 중 일부 알파벳(D, N, E) 사용](https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/) <br>
클래스 수: 3 (D, N, E) <br>
이미지 수: 클래스당 약 2000장 <br>
구조:
```
dataset/
 ├─ D/
 ├─ N/
 └─ E/
```

### 데이터 전처리 파이프라인 (OpenCV 기반)
근무표는 단순한 이미지가 아니라 표 형태이므로, OpenCV를 이용해 표 구조를 인식하고 셀 단위로 자동 분할합니다.
1. 이진화 및 노이즈 제거
2. 수평선/수직서너 검출
3. 교차점 추출 및 셀 분할
4. 각 셀 영역 Tight Crop


