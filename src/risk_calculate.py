import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


def spark_pixel(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()
    min_intensity_pixel = np.unravel_index(np.argmin(pixels), gray.shape)

    # 시각화 코드
    # image_with_min_loc = cv2.circle(image.copy(), (min_intensity_pixel[1], min_intensity_pixel[0]), 5, (0, 0, 255),
    #                                 2)  # BGR 컬러로 표시하므로 (0, 0, 255)는 빨간색을 의미합니다.
    # plt.imshow(image_with_min_loc)
    # plt.show()

    min_intensity_pixel = torch.Tensor(list(min_intensity_pixel))
    return min_intensity_pixel

a = "1201_353"
b = "1201_350"
path1 = f"output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/density_map/update1_biwi_eth_{a}_sum.png"
path2 = f'output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/density_map/update1_biwi_eth_{b}_sum.png'

distance = float(F.pairwise_distance(spark_pixel(path1), spark_pixel(path2)))
print(distance)

