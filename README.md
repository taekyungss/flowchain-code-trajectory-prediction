
# flowchain-trajectory-prediction

### 모델 설명

FlowChain 모델을 사용하여 경로 예측후, 확률 값을 기반으로 앞으로 객체가 이동할 경로의 확률을 나타내는 density map을 생성한다.


FlowChain 모델은 Conditional Continuously-indexed (CIFs) 흐름을 쌓아, 각 시점의 분포를 이전 시점으로부터 유도하여 미래 위치에 대한 확률 밀도 추정을 빠르고 정확하게 수행하는 모델이다. 

![image1.jpg](images%2Fimage01.png)

위 그림처럼 순차적 조건부 CIFs (Conditional Continuously-indexed flows)로 구성되어 

이 관찰된 궤적인 에서 인코딩된 Temporal-Social Encoder로부터 Motion Trend로 표시된 특징 벡터에 따라 조건이 지정된다. 

해당 모델은 CIFs를 사용해서 이전에 계산된 변환과 log determinant jacobian을 재사용함으로써 업데이트를 기존보다 훨씬 더 빠르고 정확하게 수행하도록 한다. 
이 접근 방식은 밀리초 단위의 업데이트를 가능하게 한다. 

따라서, 해당 모델을 통해 지게차와 사람에 대한 실시간 경로 예측이 가능하다. 


[Weight google drive](https://drive.google.com/drive/folders/1T-Chadz-1OfEzWD5DmEFPAeiKqT2oJNK?usp=sharing)

### 모델 pt값 위치
```python
output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/ckpt.pt
```

## Inference

### Step1
#### inference.py를 실행해서 density map 생성

#### run infernece.py

```python
--config_file
config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp.yml
--gpu
0
--mode
test
--visualize
```

### Step2
#### 거리 계산
#### run src/risk_calculate.py
해당 유클리디안 거리 기반으로 위험도  판단


![image1.jpg](images%2Fimage1.jpg)
