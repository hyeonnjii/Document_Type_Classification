# Document_Type_Classification
[CV / 분류] 문서 타입 이미지 분류 

## Summary

### 🛠️ 문제 정의

- **목표**: 금융, 의료, 보험 등 총 17개의 다양한 비즈니스 분야에서 대량의 문서 이미지를 식별, 자동화 처리를 가능케할 목적으로 문서 데이터 타입을 분류하는 모델 구축

### ⚙️ 수행 역할
- **라벨링 작업**: PublayNet 데이터셋과 같은 방법으로 레이아웃 정보를 포함한 데이터셋을 구축하기 위해 학습 데이터 라벨링 작업
- **OCR 라이브러리**: Tesseract 라이브러리를 사용하여 문맥에 대한 추가 정보를 도입하고자 OCR 방법론 시도
- **데이터 증강**: Albumentations, Augraphy 라이브러리를 사용하여 평가 데이터와 비슷한 방법으로 변환하여 이미지 증강


### 📈 결과 및 직무에 적용할 점
1. **이미지 증강 기술 적용**: Albumentations과 Augraphy를 활용하여 데이터셋의 다양성을 증가시키고, 평가 데이터와 유사한 품질의 학습 데이터를 구축
2. **모델 앙상블으로 성능 향상**: ResNet34, ResNet50, ResNet101 모델을 활용하여 1차 분류기를 고정하고, 각 클래스에 대해 다른 모델을 앙상블하여 성능을 극대화.


<br><br>


$${\color{gray}(※ 아래는 프로젝트의 상세 내용 입니다.)}$$
## 상세

### 📝 데이터 수집
* **데이터셋 설명**
  - 데이터는 총 17개 종의 문서로 분류되어 있으며, 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측

* **데이터 구조**
  - train 폴더에는 총 1570장의 이미지가 저장되어 있으며, 이 이미지들은 train.csv 파일에 의해 정답 클래스로 매핑
  - train.csv는 각 이미지 파일명(ID)과 해당하는 정답 클래스(target)를 제공
  - meta.csv 파일은 17개의 클래스 번호와 이에 대응하는 클래스 이름(class_name)을 포함

```
├── train                    # 이미지 데이터 폴더
│   ├── 002f99746285dfdd.jpg  # 이미지 파일 1
│   ├── 008ccd231e1fea5d.jpg  # 이미지 파일 2
│   ├── ...                   # 중간 생략
│   ├── ff8a6a251ce51c95.jpg  # 이미지 파일 1569
│   └── ffc22136f958deb1.jpg  # 이미지 파일 1570
│
├── train.csv               # 이미지 파일명과 정답 클래스 번호를 포함한 CSV 파일
│
└── meta.csv                # 클래스 번호와 클래스 이름을 포함한 메타데이터 CSV 파일

```

* 데이터 출처: Upstage 'Document Type Classification' 대회 데이터(비공개 데이터)


---
### 📊 EDA
#### 대용량 데이터 처리
