import os
import cv2
import pandas as pd
import numpy as np


# 1안의 iou값 계산 근데 보면 동일한 위치 선상에서 비교하는게 아님
# def calculate_IOU(image_path1, image_path2):
#     # 이미지 읽어오기 및 grayscale 변환
#     image1 = cv2.imread(image_path1)
#     image2 = cv2.imread(image_path2)
#     image_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     image_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
#     # 이미지 이진화
#     res, thresh1 = cv2.threshold(image_1, 195, 255, cv2.THRESH_BINARY)
#     res, thresh2 = cv2.threshold(image_2, 195, 255, cv2.THRESH_BINARY)
#
#
#     # IOU를 위한 intersection , union구하기 => 수정 필요 동일한 위치 선상에서 비교할 수있도록 코드 수정해야할듯
#     intersection = cv2.countNonZero(cv2.bitwise_and(thresh1, thresh2))
#     union = cv2.countNonZero(cv2.bitwise_or(thresh1,thresh2))
#
#     IOU = intersection / union
#     return IOU

# 2안의 iou값 계산
def calculate_IOU(image_path1, image_path2):

    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    _, thresh1 = cv2.threshold(image1, 180, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(image2, 180, 255, cv2.THRESH_BINARY)

    # 두 이미지의 모든 픽셀을 비교하여 겹치는 영역 계산
    # numpyarray에서 임계값을 기반으로 이진화된 이미지에서, 동일한 이미지에서 두 픽셀의 결과를 비교하여 겹치는 부분 저장
    # 문제 또 이렇게 하니까 흰 영역이 너무 많아서 값이 매우 작게 나옴,,, iou값이

    intersection = np.logical_and(thresh1, thresh2)
    intersection_area = np.sum(intersection)
    # print(intersection_area)

    # 두 이미지의 흰색(전체 영역) 계산
    area1 = np.sum(thresh1)
    area2 = np.sum(thresh2)

    # IOU 계산
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0

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

    for group_key, images in grouped_images.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    image_path1 = os.path.join(folder_path, images[i])
                    image_path2 = os.path.join(folder_path, images[j])
                    # print(image_path1,"/",image_path2)

                    iou = calculate_IOU(image_path1, image_path2)
                    print(iou)
                    result.append({'Image1': images[i], 'Image2': images[j], 'IOU': iou})

    # 이후, 계산 결과에 대해 csv 파일로 저장

    df = pd.DataFrame(result)
    df.to_csv("iou_results.csv", index=False)

if __name__ == "__main__":
    folder_path = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map"
    main(folder_path)
