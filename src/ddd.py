import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_7_4_sum.png"  # 첫 번째 이미지 경로
image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_8_4_sum.png"  # 두 번째 이미지 경로

image1 = cv2.imread(image_path1)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# 흑백으로 전환

res, image1_thr = cv2.threshold(image1_gray, 220, 255, cv2.THRESH_BINARY)


image2 = cv2.imread(image_path2)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
res, image2_thr = cv2.threshold(image2_gray, 220, 255, cv2.THRESH_BINARY)


intersection = cv2.countNonZero(cv2.bitwise_and(image1_thr, image2_thr))
plt.imshow(cv2.bitwise_and(image1_thr, image2_thr), cmap='gray')
plt.show()

union = cv2.countNonZero(cv2.bitwise_or(image1_thr,image2_thr))
plt.imshow(cv2.bitwise_or(image1_thr, image2_thr), cmap='gray')
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
