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
- 1, 14와 같이 특정 클래스의 이미지 개수가 적은 것을 확인
<div align='center'>
  
  ![image](https://github.com/user-attachments/assets/1800eee3-0b8a-4edf-b464-8b67be004a4e)

</div>

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


---
### 🧠 모델링
#### 평가 지표
- [Macro F1 Score](https://www.linkedin.com/pulse/understanding-confusion-matrix-tanvi-mittal/)
  * Macro F1 score는 multi classification을 위한 평가 지표로 클래스 별로 계산된 F1 score를 단순 평균한 지표
  

<div align='center'>
  
  ![image](https://quicklatex.com/cache3/ec/ql_37c43c91b217bd926d84481131e1bcec_l3.png)

</div>

    * N: 클래스의 수
    * F1_i: 각 클래스의 F1 점수. F1 점수는 정밀도와 재현율의 조화 평균으로 계산
    


#### 모델링 전략
- **Class 별 Augmentation**: 평가 데이터를 모방한 학습 데이터를 구축
- **2 Step Classifier**: Car, License, Paper 먼저 분류 후, 각 그룹내에서 실제 Type을 구분하는 2 Step Classifier 학습


#### 모델 선정

- Transformer 계열 vs. **CNN 계열**✔
  - 문서 데이터의 낮은 해상도와 데이터 수의 부족으로 인해 Transformer 계열 모델이 제대로 학습되지 않음을 확인
  - **문서 데이터, 모델의 특성을 고려하여 CNN 계열의 ResNet, EfficientNet 사용**
    * **데이터 특성**: Transformer 모델은 일반적으로 고해상도 이미지나 대량의 데이터에서 뛰어난 성능을 발휘. 그러나 문서 데이터는 낮은 해상도와 데이터 수의 부족으로 인해 모델이 필요한 정보를 충분히 학습하기 어려웠을 것
    * **모델의 복잡성**: Transformer는 매우 복잡한 구조를 가지고 있어, 학습에 필요한 데이터의 양이 많음. 각 클래스에 대한 데이터가 적은 경우 일반화 성능 떨어졌을 것


#### 모델 별 성능 평가

<div align='center'>

  | Model     | 2 step classifier | Augmentation | F1 score |
  |-----------|-------------------|--------------|----------|
  | Resnet34  | X                 | X            | 0.1967   |
  | Resnet34  | O                 | X            | 0.6226   |
  | ResNet18  | O                 | X            | 0.7985   |
  | ResNet18  | O                 | O            | 0.8468   |
  | ResNet101 | O                 | O            | 0.8650   |

</div> 

```
- Augmentation 을 적용한 모델 학습 결과가 대체적으로 더 좋게 나온 것을 확인 
```
<div align='center'>
  
  ![image](https://github.com/user-attachments/assets/db397aae-6c18-4ff9-b99b-35f56bc74206)

</div>

```
- Resnet34, 50, 101 을 기본 파라미터로 학습한 wandb 결과
- 결과적으로 1차 분류기를 고정 후. 각 클래스의 모델을 다르게 하여 앙상블 한 모델이 가장 성능이 좋았음
- 1차 분류기에서는 resnet 34의 성능이, paper 카테고리에서는 resnet101의 성능이 가장 좋았음
```

---
### 🎯 결과 및 기대효과
- **데이터 증강 효과**: Albumentations와 Augraphy를 통한 이미지 증강 기법이 데이터 다양성을 높이고, 모델의 일반화 성능을 향상시키는 데 기여
- **비즈니스 적용 가능성**: 본 모델은 금융, 의료, 보험 등 다양한 산업 분야에서 실제 대량의 문서 이미지를 자동으로 분류할 수 있는 기반을 마련. 이는 기업의 문서 처리 효율성을 높이는데 기여
- **OCR, Super resolution 기법**: 해당 프로젝트에서는 이러한 방법론들이 성능 향상으로 이어지지는 않았지만, 향후 문서 데이터에 대하여 이러한 기법을 적용해볼 수 있음음

 




