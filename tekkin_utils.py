import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import glob
# tets
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import defaultdict
"""
For prepare_heatmap
"""

from matplotlib import pyplot as plt

def draw_bounding_boxes(image_path, boxes):
    # 画像を読み込む
    image = cv2.imread(image_path)

    # 画像が正しく読み込まれたかチェック
    if image is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return

    # BGRからRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # バウンディングボックスを描画
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 7)

    # 画像を表示
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def xml_to_df(anotated_dir):
    xml_list = []
    file_list = os.listdir(anotated_dir)
    for i, file in enumerate(file_list):
        print(f'\r {i+1}/{len(file_list)}', end='')
        base, ext = os.path.splitext(file)
        if ext == '.xml':
            tree = ET.parse(anotated_dir + file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size').find('width').text),
                         int(root.find('size').find('height').text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'classname', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def putGaussianMaps(center, accumulate_confid_map, params_transform):
    '''ガウスマップに変換する'''
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    stride = params_transform['stride']
    sigma = params_transform['sigma']

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map

"""
For point_torch
"""
import albumentations as albu
def get_training_augmentation(mask_size=1280):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=mask_size, min_width=mask_size, always_apply=True, border_mode=0),
        albu.RandomCrop(height=mask_size, width=mask_size, always_apply=True),

#         albu.IAAAdditiveGaussianNoise(p=0.2),
#         albu.IAAPerspective(p=0.5),

#         albu.OneOf(
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.IAASharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.9,
#         ),

#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1),
#                 albu.HueSaturationValue(p=1),
#             ],
#             p=0.9,
#         ),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def to_tensor_mask(x, **kwargs):
    x = x[np.newaxis, :, :]
    return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)

class Loss(nn.Module):
    __name__ = 'loss'

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        loss = F.mse_loss(input, target, reduction='mean')

        return loss


class Dataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids if
                           image_id.split('.')[1] == 'PNG' or image_id.split('.')[1] == 'jpg']
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids if
                          image_id.split('.')[1] == 'PNG' or image_id.split('.')[1] == 'jpg']

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0) / 255

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)
