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

import cv2
import numpy as np

def calculate_IOU(image_path1, image_path2):

    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    _, thresh1 = cv2.threshold(image1, 240, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(image2, 240, 255, cv2.THRESH_BINARY)


    intersection = np.logical_and(thresh1, thresh2)
    intersection_area = np.sum(intersection)
    print(intersection_area)


    # union
    union = np.logical_or(thresh1, thresh2)
    union_area = np.sum(union)
    print(union_area)

    union_image  = np.zeros_like(image1)
    union_image[union] = 255
    cv2.imshow("union_image", union_image)
    cv2.waitKey(0)

    iou = intersection_area / union_area

    return iou

if __name__ == "__main__":
    image_path1 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_10_4_sum.png"  # 첫 번째 이미지 경로
    image_path2 = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map/update1_tmp_10_5_sum.png"  # 두 번째 이미지 경로

    iou = calculate_IOU(image_path1, image_path2)
    print("IOU:", iou)
