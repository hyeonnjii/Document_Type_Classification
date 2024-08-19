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
### 문서 이미지 이해를 바탕으로 한 가설 수립

#### 가설 1. Document Layout Classification, LayoutParser 문제로 해결할 수 있을 것이다.
- [PublayNet](https://github.com/ibm-aur-nlp/PubLayNet)와 같은 형태로 학습 데이터를 만드는 접근 방법
- [labelImg](https://github.com/HumanSignal/labelImg) 라이브러리를 사용하여 이미지 라벨링 작업 수행
- 하지만, 차량 번호판, 계기판 등 문서라고 정의하기 애매한 클래스 이미지 존재
![image](https://github.com/user-attachments/assets/fb321e82-4e63-4c40-889a-76dfdfc142f6)


#### 가설 2. 비슷한 문서 형태의 클래스를 1차적으로 대분류한 후, 각 분류에 맞는 2단계 모델링 학습을 적용하면 성능이 향상 것이다.
- Car, License, Paper 세 카테고리로 1차 분류
- 보다 쉽게 해결할 수 있는 문제부터 접근한 후, 세부 분류를 통해 문제의 난이도를 낮추는 전략
- 다만, Car(2), License(4), Paper(11) 로 카테고리별 데이터 수의 불균형을 고려해야 함
- 또한, Paper 의 경우 사람이 판단하기에도 애매한 데이터가 다수 존재하므로 이에 대한 해결책이 필요해 보임
![image](https://github.com/user-attachments/assets/77d9f2b4-8b16-4247-9019-88575ca14b8d)


#### 가설 3. OCR을 통해 문맥에 대한 글자 정보를 더해준다면 성능이 향상될 것이다. 
- Paper 카테고리의 데이터의 경우 문서 내용을 이해하는 것이 보다 중요하다고 판단
- 비교적 정상적인 상태의 문서의 경우 OCR 결과가 유의미해 보이지만 뒤집히거나 품질이 좋지 않은 경우 정상적인 인식이 어려운 문제 발생
![image](https://github.com/user-attachments/assets/4cfafbee-85fe-4196-ad22-7d9de9ea1d80)

#### 가설 4. 평가 데이터와 비슷한 형태로 문서 데이터의 변형을 가한 데이터로 학습한다면 성능이 향상될 것이다.
- 주어진 평가 데이터에 사용된 증강 기법을 추측하여 사용
- Albumentations, Augraphy 라이브러리를 사용


![image](https://github.com/user-attachments/assets/e0a38553-64d0-49af-a559-44664c524402)


---
### ⚒ Feature Engineering
#### 이미지 데이터 증강
- 평가 데이터에 적용된 효과들을 최대한 비슷하게 구현하기 위해 이미지 증강 라이브러리인 [Albumentations](https://github.com/albumentations-team/albumentations) 과 [Augraphy](https://augraphy.readthedocs.io/en/latest/doc/source/list_of_augmentations.html) 을 사용
  - Albumentations은 이미지 텐서 차원, 값을 조작하여 보다 일반적인 이미지 변환에 사용되는 기법을 제공
  - Augraphy는 주로 문서 이미지의 잉크, 종이 부분을 구분하고 각 부분에 맞게 실제 문서에 일어날 수 있는 변형 기법을 제공
 
<div align="center">
      
  ![image](https://github.com/user-attachments/assets/bfcb85b4-7aa3-402c-a872-812a72649887)

</div>

  - 주로 Albumentations의 Mixup(), Flip(), Rotate() 등의 기법과, Augraphy의 noiseTexturize(), VoronoiTessellation() 등의 기법을 사용

<div align="center">
  
  ![image](https://github.com/user-attachments/assets/10362fc5-47d8-447e-b2c3-1ff803e5962c)

 </div>

 #### 이미지 화질 개선
 - CNN기반의 [SRCNN](https://github.com/yjn870/SRCNN-pytorch), GAN 기반의 [SRGAN](https://github.com/Lornatang/SRGAN-PyTorch)을 이용해 전체적인 이미지 데이터들의 화질 개선을 통해 성능 향상을 노력함
 - 해당 super resolution 기법을 적용하기에는 비용적인 부분의 한계로 적용까지는 진행되지 못함

<div align='center'>

![image](https://github.com/user-attachments/assets/046a62ed-34ec-438a-b0a2-b5f86112efeb)

</div>

 




