
# flowchain-trajectory-prediction


[Weight google drive](https://drive.google.com/drive/folders/1T-Chadz-1OfEzWD5DmEFPAeiKqT2oJNK?usp=sharing)

### 모델 pt값 위치
output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/ckpt.pt

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
