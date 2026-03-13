from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def sar32bit2RGB(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg / nimg.max() * 255
    nimg_8 = nimg.astype(np.uint8)
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img

import numpy as np
from PIL import Image

def mysar32bit2rgb(img, low_percent=5, high_percent=100):
    """
    将单通道 32 位 SAR 图像转换为 RGB 图像，使用百分位线性拉伸增强对比度
    :param img: 输入图像 (numpy array)，单通道，dtype 可以为 float32 或其他数值类型
    :param low_percent: 下百分位，默认值为 5
    :param high_percent: 上百分位，默认值为 100
    :return: PIL.Image 对象 (RGB 模式)
    """

    def linear_stretch(channel_img, low, high):
        """
        对单通道图像进行基于百分位的线性拉伸
        :param channel_img: 单通道图像 (numpy array)
        :param low: 下百分位
        :param high: 上百分位
        :return: 线性拉伸后的 uint8 图像
        """
        c_min = np.percentile(channel_img, low)
        c_max = np.percentile(channel_img, high)
        out = (channel_img - c_min) / (c_max - c_min + 1e-8) * 255.0  # 加小数防除零
        return np.clip(out, 0, 255).astype(np.uint8)

    # 转换为 numpy 数组并确保是 float32 类型
    nimg = np.array(img, dtype=np.float32)

    # 执行线性拉伸
    stretched = linear_stretch(nimg, low_percent, high_percent)

    # 将单通道图像扩展为三通道（灰度图 → RGB）
    rgb_img = np.stack([stretched]*3, axis=-1)  # shape: (H, W, 3)

    # 转换为 PIL.Image
    pil_img = Image.fromarray(rgb_img, mode='RGB')

    return pil_img



class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train is not None:
            num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")




class QueryAddDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform
        self.pair = pair
    def __len__(self):
        return len(self.dataset)

    def get_image(self, img_path): 
        img = read_image(img_path).convert("RGB")
        img_size = img.size
        img_size = [img_size[0] * 0.75, img_size[1] * 0.75]
        img_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427 - 0.461) / 0.031, img_size[1] / img_size[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size


    def __getitem__(self, index):
        if self.pair:
            imgs = []
            for img in self.dataset[index]:
                img_path, pid, camid = img
                im, img_size = self.get_image(img_path)
                imgs.append((im, pid, camid, img_path.split("/")[-1], img_size))
            return imgs
        else:
            img_path, pid, camid, trackid = self.dataset[index]
            img, img_size = self.get_image(img_path)
            return img, pid, camid, trackid, img_path.split("/")[-1], img_size


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform

        self.pair = pair

    def __len__(self):
        return len(self.dataset)

    def get_image(self, img_path):
        if img_path.endswith("SAR.tif"):
            img = read_image(img_path)
            # img = sar32bit2RGB(img)
            img = mysar32bit2rgb(img)
            img_size = img.size
        else:
            img = read_image(img_path).convert("RGB")
            img_size = img.size
            img_size = [img_size[0] * 0.75, img_size[1] * 0.75]
        img_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427 - 0.461) / 0.031, img_size[1] / img_size[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size

    def __getitem__(self, index):
        if self.pair:
            imgs = []
            for img in self.dataset[index]:
                img_path, pid, camid = img
                im, img_size = self.get_image(img_path)
                imgs.append((im, pid, camid, img_path.split("/")[-1], img_size))
            return imgs
        else:
            img_path, pid, camid, trackid = self.dataset[index]
            img, img_size = self.get_image(img_path)
            return img, pid, camid, trackid, img_path.split("/")[-1], img_size
