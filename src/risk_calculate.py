import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


# 저장된 density map에서 픽셀이 튀는 부분(가장 진한 부분)을 찾고 그것 끼리의 거리를 측정하는 코드

def spark_pixel(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()
    min_intensity_pixel = np.unravel_index(np.argmin(pixels), gray.shape)


    min_intensity_pixel = torch.Tensor(list(min_intensity_pixel))
    return min_intensity_pixel

# 각각 iamge pair 입력
path1 = f"첫번째 image"
path2 = f"두번째 image"

distance = float(F.pairwise_distance(spark_pixel(path1), spark_pixel(path2)))
print(distance)




