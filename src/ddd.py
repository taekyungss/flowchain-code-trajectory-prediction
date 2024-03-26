# import cv2
#
#
# def IOU(image_path1, image_path2):
#     # 이미지 읽어오기 및 grayscale 변환
#     image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#     # cv2.imwrite("o1.jpg", image1)
#     # cv2.imwrite("o2.jpg", image2)
#     # 이미지 이진화
#     ret1, thresh1 = cv2.threshold(image1, 180, 255, cv2.THRESH_BINARY)
#     ret2, thresh2 = cv2.threshold(image2, 180, 255, cv2.THRESH_BINARY)
#     intersection = cv2.countNonZero(cv2.bitwise_and(thresh1, thresh2))
#     union = cv2.countNonZero(cv2.bitwise_or(thresh1,thresh2))
#
#     IOU = intersection / union
#     # print(intersection, union, IOU)
#     return IOU
#
#
# if __name__ == "__main__":
#     image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/visualize/density_map/update1_biwi_eth_112_11_sum.png"  # Update this with the path to your first image
#     image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/visualize/density_map/update1_biwi_eth_112_12_sum.png"  # Update this with the path to your second image
#     print(IOU(image_path1, image_path2))



# import cv2
# import numpy as np
#
# def calculate_IOU(image1, image2):
#     # 이미지 이진화
#     _, thresh1 = cv2.threshold(image1, 180, 255, cv2.THRESH_BINARY)
#     _, thresh2 = cv2.threshold(image2, 180, 255, cv2.THRESH_BINARY)
#
#     # 컨투어 찾기
#     contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 두 이미지의 컨투어 영역 계산
#     area1 = cv2.contourArea(contours1[0]) if contours1 else 0
#     area2 = cv2.contourArea(contours2[0]) if contours2 else 0
#
#     # 겹치는 영역 계산
#     intersection_area = 0
#     if area1 and area2:
#         intersection = cv2.bitwise_and(thresh1, thresh2)
#         intersection_contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         intersection_area = cv2.contourArea(intersection_contours[0]) if intersection_contours else 0
#
#     # IOU 계산
#     union_area = area1 + area2 - intersection_area
#     iou = intersection_area / union_area if union_area != 0 else 0
#
#     return iou
#
# if __name__ == "__main__":
#     image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/visualize/density_map/update1_biwi_eth_112_11_sum.png"  # 첫 번째 이미지 경로
#     image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/visualize/density_map/update1_biwi_eth_112_12_sum.png"  # 두 번째 이미지 경로
#
#     image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#
#     iou = calculate_IOU(image1, image2)
#     print("IOU:", iou)


''' 가장 최근 코드  20240326 주석처리'''

import cv2
import numpy as np


# def calculate_IOU(image_path1, image_path2):
#
#     image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#     cv2.imshow('image1', image1)
#     cv2.imshow('image2', image2)
#
#     _, thresh1 = cv2.threshold(image1, 220, 255, cv2.THRESH_BINARY)
#     _, thresh2 = cv2.threshold(image2, 220, 255, cv2.THRESH_BINARY)
#
#     # 두 이미지의 모든 픽셀을 비교하여 겹치는 영역 계산
#     # numpyarray에서 임계값을 기반으로 이진화된 이미지에서, 동일한 이미지에서 두 픽셀의 결과를 비교하여 겹치는 부분 저장
#     # 문제 또 이렇게 하니까 흰 영역이 너무 많아서 값이 매우 작게 나옴,,, iou값이
#
#     intersection = np.logical_and(thresh1, thresh2)
#     intersection_area = np.sum(intersection)
#     print(intersection_area)
#
#     # # 시각화 용
#     intersection_image = np.zeros_like(image1)
#     intersection_image[intersection] = 255
#     cv2.imshow("thresh1", thresh1)
#     cv2.imshow("thresh2", thresh2)
#     cv2.imshow("union_image", intersection_image)
#
#     # union
#     union = np.logical_or(thresh1, thresh2)
#     union_area = np.sum(union)
#     print(union_area)
#
#
#     union_image  = np.zeros_like(image1)
#     union_image[union] = 255
#     cv2.imshow("intersection", union_image)
#     cv2.waitKey(0)
#
#     iou = intersection_area / union_area
#
#     return iou

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_7_4_sum.png"  # 첫 번째 이미지 경로
image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_8_4_sum.png"  # 두 번째 이미지 경로

grd_truth = cv2.imread(image_path1)
# plt.imshow(grd_truth)
# plt.show()
# Ground truth 이미지를 불러옵니다.

grd_truth_gray = cv2.cvtColor(grd_truth, cv2.COLOR_BGR2GRAY)
# 흑백으로 전환합니다. (왜냐면, cv2.threshold()를 사용해야하기 때문입니다. cv2.threshold()를 사용하는 이유는, bit_wise연산을 하려면 0,1로만 이루어진 이미지를 구해야하기 때문입니다.)

res, grd_truth_thr = cv2.threshold(grd_truth_gray, 220, 255, cv2.THRESH_BINARY)
# grd_truth_thr은 0,1로만 이루어진 numpy.ndarray입니다. 0은 해당 픽셀이 검정색 픽셀이라는 뜻이고, 1은 해당 픽셀이 흰색 픽셀이라는 뜻입니다.
plt.imshow(grd_truth_thr)
plt.show()


masked = cv2.imread(image_path2)
masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
res, masked_thr = cv2.threshold(masked_gray, 220, 255, cv2.THRESH_BINARY)
# 마스크 처리된(segment한 이미지)도 마찬가지로 흑백으로 전환시켜줍니다.
plt.imshow(masked)
plt.show()

intersection = cv2.countNonZero(cv2.bitwise_and(grd_truth_thr, masked_thr))
# and연산을 해서, 교집합(같은 영역을 비교해서 해당 픽셀에서 둘 다 1이면 교집합이기 때문에 1로 연산합니다.)을 구합니다.
# 그리고 countNonZero로 픽셀이 0이 아닌 것만(검정 픽셀) 계산합니다. 그것이 면적입니다.
plt.imshow(cv2.bitwise_and(grd_truth_thr, masked_thr), cmap='gray')
plt.show()

union = cv2.countNonZero(cv2.bitwise_or(grd_truth_thr,masked_thr))
# or연산을 해서, 합집합(같은 영역을 비교해서 해당 픽셀에서 어느 하나라도 1이면 1로 처리해서 합집합을 구현합니다.)을 구합니다.
# 그리고 countNonZero로 픽셀이 0이 아닌 것만(검정 픽셀) 계산합니다. 그것이 면적입니다.
plt.imshow(cv2.bitwise_or(grd_truth_thr, masked_thr), cmap='gray')
plt.show()

if intersection == 0 and union == 0:
    IoU = 0
else:
    IoU = intersection / union

print(intersection, union, IoU)


# if __name__ == "__main__":
#     image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_7_4_sum.png"  # 첫 번째 이미지 경로
#     image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_8_4_sum.png"  # 두 번째 이미지 경로
#
#     iou = calculate_IOU(image_path1, image_path2)
#     print("IOU:", iou)
