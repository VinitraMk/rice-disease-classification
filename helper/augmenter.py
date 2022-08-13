from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
import PIL.Image

from helper.utils import get_config

class Augmenter:
    old_train = None
    old_train_rgn = None
    flip_transform = None

    def __init__(self, train, train_rgn):
        self.old_train = train
        self.old_train_rgn = train_rgn
        self.resize_transform = transforms.Resize((800, 800))
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.rotate_90transform = transforms.RandomRotation(90)
        self.rotate_180transform = transforms.RandomRotation(180)
        self.rotate_270transform = transforms.RandomRotation(270)
        self.brightness_increase_transform = lambda img: F.adjust_brightness(img, 1.5)
        self.brightness_decrease_transform = lambda img: F.adjust_brightness(img, 0.5)
        self.contrast_transform = transforms.ColorJitter(contrast=2)
        self.saturation_transform = transforms.ColorJitter(saturation=2)
        self.hue_transform = transforms.ColorJitter(hue=0.5)

    def __augment_data(self, data, is_rgn = False):
        new_list = []
        for img_el in data:
            #img_tensor = transforms.ToTensor(cv2.imread(img_el['data_path']))
            img_tensor = PIL.Image.open(img_el['data_path'])
            if is_rgn:
                img_tensor = self.resize_transform(img_tensor)
            img_label = img_el['label']
            img_id = img_el['Image_id']
            flip_img = self.flip_transform(img_tensor)
            new_list.append({ 'image_tensor': flip_img, 'label': img_label, 'Image_id': f'{img_id}_flip', 'image_type': 'image_tensor' })
            img_90 = self.rotate_90transform(img_tensor)
            new_list.append({ 'image_tensor': img_90, 'label': img_label, 'Image_id': f'{img_id}_rotate90', 'image_type': 'image_tensor' })
            img_180 = self.rotate_180transform(img_tensor)
            new_list.append({ 'image_tensor': img_180, 'label': img_label, 'Image_id': f'{img_id}_rotate180', 'image_type': 'image_tensor' })
            img_270 = self.rotate_270transform(img_tensor)
            new_list.append({ 'image_tensor': img_270, 'label': img_label, 'Image_id': f'{img_id}_rotate270', 'image_type': 'image_tensor' })
            img_brightness_inc = self.brightness_increase_transform(img_tensor)
            new_list.append({ 'image_tensor': img_brightness_inc, 'label': img_label, 'Image_id': f'{img_id}_brinc', 'image_type': 'image_tensor' })
            img_brightness_dec = self.brightness_decrease_transform(img_tensor)
            new_list.append({ 'image_tensor': img_brightness_dec, 'label': img_label, 'Image_id': f'{img_id}_brdec', 'image_type': 'image_tensor' })
            img_contrast = self.contrast_transform(img_tensor)
            new_list.append({ 'image_tensor': img_contrast, 'label': img_label, 'Image_id': f'{img_id}_contrast', 'image_type': 'image_tensor' })
            img_sat = self.saturation_transform(img_tensor)
            new_list.append({ 'image_tensor': img_sat, 'label': img_label, 'Image_id': f'{img_id}_sat', 'image_type': 'image_tensor' })
            img_hue = self.hue_transform(img_tensor)
            new_list.append({ 'image_tensor': img_hue, 'label': img_label, 'Image_id': f'{img_id}_hue', 'image_type': 'image_tensor' })
        return new_list


    def augment_data(self):
        config = get_config()
        new_list = self.__augment_data(self.old_train)
        new_list = self.old_train + new_list
        new_list_rgn = self.__augment_data(self.old_train_rgn, True)
        new_list_rgn = self.old_train_rgn + new_list_rgn
        return new_list, new_list_rgn
