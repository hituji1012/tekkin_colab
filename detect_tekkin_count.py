import torch
import torchvision
import segmentation_models_pytorch as smp
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy import ndimage
import numpy as np
import cv2

from matplotlib import cm
from PIL import Image

# ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
# DECODER = 'psp'
class TekkinCount():
    def __init__(self, encoder='resnet34', decoder='psp', model_path='model'):
        self.encoder = encoder
        self.decoder = decoder
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, ENCODER_WEIGHTS)
        self.model = torch.load(f'{model_path}/{self.decoder}_{self.encoder}.pth', map_location=torch.device(DEVICE)).eval()
        
    def find_peaks(self, img):
        """
        Given a (grayscale) image, find local maxima whose value is above a given
        threshold (param['thre1'])
        :param img: Input image (2d array) where we want to find peaks
        :return: 2d np.array containing the [x,y] coordinates of each peak found
        in the image
        """
        struct = generate_binary_structure(2, 1)
        new_struct = np.zeros((13, 13))
        new_struct[6, 6] = 1
        new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
        new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
        new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
#         new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
#         new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
#         new_struct = ndimage.binary_dilation(new_struct, structure=struct).astype(new_struct.dtype)
        #     new_struct
        #     print(new_struct)
        peaks_binary = (maximum_filter(img, footprint=new_struct) == img) * (img > 0.4)
        # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
        # x]...]
        return np.array(np.nonzero(peaks_binary)[::-1]).T

    def count(self, img):

        # for fn in file_list:
        image_src = cv2.resize(img, (1280, 1280))
        image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        image = self.preprocessing_fn(image)
        image = image.transpose(2, 0, 1).astype('float32')
        image = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        predict = self.model(image)
        predict = predict.detach().cpu().numpy()[0]

        # heat_map = Image.fromarray(np.uint8(cm.jet(predict[0,:,:])*255))
        # heat_map = np.asarray(heat_map.convert('RGB'))
        # blend_img = cv2.addWeighted(image_src, 0.5, heat_map, 0.5, 0)

        # fig = plt.figure(figsize=(20, 10))
        # plt.subplot(1,2,1)
        # plt.imshow(blend_img)
        # ピーク数取得
        max_img = self.find_peaks(predict[0, :, :])
        print(f'鉄筋数 {max_img.shape[0]}')
        max_img[:,0] = max_img[:,0] * (img.shape[1]/1280)
        max_img[:,1] = max_img[:,1] * (img.shape[0]/1280)
        return max_img.shape[0], max_img
        # max_img = max_img.tolist()
        # heat_map = np.zeros((mask_size, mask_size))
        # for p in max_img:
        #     center = [p[0], p[1]]
        #     gaussian_map = heat_map
        #     heat_map = putGaussianMaps(
        #         center, gaussian_map, params_transform)

        # heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
        # heat_map = np.asarray(heat_map.convert('RGB'))
        # plt.subplot(1,2,2)
        # plt.imshow(heat_map)
        # fig.savefig(f'heat_{os.path.basename(fn)}')