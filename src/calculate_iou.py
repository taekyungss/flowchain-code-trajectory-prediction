import os
import cv2
import pandas as pd

def calculate_IOU(image_path1, image_path2):
    # 이미지 읽어오기 및 grayscale 변환
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 이미지 이진화
    ret1, thresh1 = cv2.threshold(image1, 195, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(image2, 195, 255, cv2.THRESH_BINARY)

    # IOU를 위한 intersection , union구하기 => 수정 필요 동일한 위치 선상에서 비교할 수있도록 코드 수정해야할듯
    intersection = cv2.countNonZero(cv2.bitwise_and(thresh1, thresh2))
    union = cv2.countNonZero(cv2.bitwise_or(thresh1,thresh2))

    IOU = intersection / union
    return IOU

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

    for group_key, images in grouped_images.items():
        if len(images) >= 2:
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    image_path1 = os.path.join(folder_path, images[i])
                    image_path2 = os.path.join(folder_path, images[j])
                    print(image_path1,"/",image_path2)

                    #calculate IOU
                    iou = calculate_IOU(image_path1, image_path2)
                    print(iou)
                    result.append({'Image1': images[i], 'Image2': images[j], 'IOU': iou})

    df = pd.DataFrame(result)
    df.to_csv("iou_results.csv", index=False)

if __name__ == "__main__":
    folder_path = "output/config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth/visualize/density_map"
    main(folder_path)
