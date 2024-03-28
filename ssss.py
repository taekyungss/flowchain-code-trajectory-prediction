import os
import cv2
import numpy as np
import pandas as pd

def extract_number_from_filename(filename):
    numbers = [int(s) for s in filename.split('_') if s.isdigit()]
    return numbers

def calculate_IOU(image1, image2):
    # 이미지 이진화
    _, thresh1 = cv2.threshold(image1, 220, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(image2, 220, 255, cv2.THRESH_BINARY)

    # 두 이미지의 모든 픽셀을 비교하여 겹치는 영역 계산
    intersection = np.logical_and(thresh1, thresh2)
    intersection_area = np.sum(intersection)

    # 두 이미지의 흰색(전체 영역) 계산
    area1 = np.sum(thresh1)
    area2 = np.sum(thresh2)

    # IOU 계산
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0

    return iou

def compare_group_images(grouped_images):
    result = []
    for group_key1, images1 in grouped_images.items():
        for group_key2, images2 in grouped_images.items():
            if group_key1 != group_key2:  # 서로 다른 그룹인 경우에만 비교
                for image_path1 in images1:
                    for image_path2 in images2:
                        numbers1 = extract_number_from_filename(image_path1)
                        numbers2 = extract_number_from_filename(image_path2)
                        if numbers1[1] == numbers2[1]:  # 동일한 위치에 있는 경우에만 비교
                            image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
                            image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
                            iou = calculate_IOU(image1, image2)
                            result.append({'Image1': image_path1, 'Image2': image_path2, 'IOU': iou})


    return result

def main(folder_path):
    grouped_images = {}

    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    for png_file in png_files:
        numbers = extract_number_from_filename(png_file)
        group_key = numbers[1]
        if group_key not in grouped_images:
            grouped_images[group_key] = []
        grouped_images[group_key].append(os.path.join(folder_path, png_file))

    result = compare_group_images(grouped_images)

    # 결과를 CSV 파일로 저장
    df = pd.DataFrame(result)
    df.to_csv('iou_results.csv', index=False)

if __name__ == "__main__":
    folder_path = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/tmp/visualize/density_map"
    main(folder_path)
