import os
import cv2
import pandas as pd
import numpy as np


# 1안의 iou값 계산 근데 보면 동일한 위치 선상에서 비교하는게 아님
# def calculate_IOU(image_path1, image_path2):
#     image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#     # 이미지 이진화
#     _, thresh1 = cv2.threshold(image1, 230, 255, cv2.THRESH_BINARY)
#     _, thresh2 = cv2.threshold(image2, 230, 255, cv2.THRESH_BINARY)
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

# 2안의 iou값 계산
def calculate_IOU(image_path1, image_path2):

    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('image1', image1)
    # cv2.imshow('image2', image2)

    _, thresh1 = cv2.threshold(image1, 240, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(image2, 240, 255, cv2.THRESH_BINARY)

    # 두 이미지의 모든 픽셀을 비교하여 겹치는 영역 계산
    # numpyarray에서 임계값을 기반으로 이진화된 이미지에서, 동일한 이미지에서 두 픽셀의 결과를 비교하여 겹치는 부분 저장
    # 문제 또 이렇게 하니까 흰 영역이 너무 많아서 값이 매우 작게 나옴,,, iou값이

    intersection = np.logical_and(thresh1, thresh2)
    intersection_area = np.sum(intersection)

    # # 시각화 용
    # intersection_image = np.zeros_like(image1)
    # intersection_image[intersection] = 255
    # cv2.imshow("thresh1", thresh1)
    # cv2.imshow("thresh2", thresh2)
    # cv2.imshow("intersection", intersection_image)
    # cv2.waitKey(0)

    # union
    union = np.logical_or(thresh1, thresh2)
    union_area = np.sum(union)

    union_image = np.zeros_like(image1)
    union_image[union] = 255
    # cv2.imshow("union_image", union_image)
    # cv2.waitKey(0)

    iou = intersection_area / union_area

    return iou

# file이름 에서 track_id별로 비교하기 위해서 숫자 뽑아내기
def extract_number_from_filename(filename):
    numbers = [int(s) for s in filename.split('_') if s.isdigit()]
    return numbers


def main(folder_path):
    result = []
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    grouped_images = {}
    for png_file in png_files:
        numbers = extract_number_from_filename(png_file)
        group_key = numbers[1]
        if group_key not in grouped_images:
            grouped_images[group_key] = []
        grouped_images[group_key].append(png_file)

    # 여기서 이미지들에 대해 groupkey 즉, filename안에 있는 frame(?)이 같은데 track_id 가 다른 경우 매칭하도록 만들기
    # 해당 매칭된 데이터들에 대해서는 같이 iou계산하도록

    # EDIT : 수정해야할 사항 : frame 단위마다 density map 저장 시키고, 해당 프레임 내에서 다른 객체끼리 비교해서
    # 해당 부분에 대한 경우의 수를 계산하도록 할것 우리가 생각했던 첫번째 index는 앞으로 예측할 frame, 그리고 그 뒤에 index는 그 이후 track_id가 맞음
    # + yolo랑 이거 IOU겹치는 부분에 대해 시각화해서 교수님 설명자료 어여 만들기




    # for group_key, images in grouped_images.items():
    #     if len(images) >= 2:
    #         for i in range(len(images)):
    #             for j in range(i + 1, len(images)):
    #                 image_path1 = os.path.join(folder_path, images[i])
    #                 image_path2 = os.path.join(folder_path, images[j])
    #                 print(image_path1,"/",image_path2)
    #
    #
    #                 iou = calculate_IOU(image_path1, image_path2)
    #                 print(iou)
    #                 result.append({'Image1': images[i], 'Image2': images[j], 'IOU': iou})

    # 이후, 계산 결과에 대해 csv 파일로 저장

    df = pd.DataFrame(result)
    df.to_csv("iou_results.csv", index=False)

if __name__ == "__main__":
    folder_path = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map"
    result = main(folder_path)
    print(result)
