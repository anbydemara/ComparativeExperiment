#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ComparativeExperiment 
@File    ：crop.py
@IDE     ：PyCharm 
@Author  ：一只快乐鸭
@Date    ：2025/4/21 17:16 
"""
import os
from PIL import Image

def process_image(root, file):
    """
    处理单张图像的函数。
    你可以在这里添加任何你需要的图像处理逻辑。
    """
    image_path = os.path.join(root, file)
    try:
        with Image.open(image_path) as img:
            # 定义裁剪区域 (左, 上, 右, 下)
            crop_area = (900, 500, 900+350, 500+550)  # 示例坐标
            # 裁剪图片
            cropped_img = img.crop(crop_area)
            # 保存裁剪后的图片
            cropped_img.save(f'{root}/crop_{file}')

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_folder(folder_path):
    """
    遍历文件夹并处理所有图像。
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                process_image(root, file)


if __name__ == '__main__':
    folder_path = './'
    # process_folder(folder_path)
    # process_image('./', 'groud_truth_PaviaC.png')
    process_image('./', 'classification_result_PaviaC.png')