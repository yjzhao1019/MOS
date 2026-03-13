# encoding: utf-8
"""
@author:  alioth
"""

import glob, cv2
import os.path as osp
from .bases import BaseImageDataset


class Pretrain(BaseImageDataset):
    dataset_dir = "OptiSar_Pair"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super(Pretrain, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir
        self.pid_begin = pid_begin

        train, train_pair, self.pid_begin = self._process_dir_train(self.train_dir, relabel=True)

        if verbose:
            print("=> Pretrain Dataset loaded")
            if train_pair is not None:
                print("Number of RGB-SAR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair

        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams = self.get_imagedata_info_pair(self.train_pair)

    def get_imagedata_info_pair(self, data):
        pids, cams = [], []
        for img in data:
            for _, pid, camid in img:
                pids += [pid]
                cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*/*"))

        RGB_paths = [i for i in img_paths if i.endswith("RGB.tif") or i.endswith("RGB.png")]
        pid2sar = {}

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split("/")[-1].split("_")[1])
            pid_container.add(pid)
            if img_path.endswith(f"SAR.tif") or img_path.endswith(f"SAR.png"):
                if pid not in pid2sar:
                    pid2sar[pid] = [img_path]
                else:
                    pid2sar[pid].append(img_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid = int(img_path.split("/")[-1].split("_")[1])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split("/")[-1].split("_")[-1].split(".")[0] == "RGB" else 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid))

        dataset_pair = []
        for img_path in sorted(RGB_paths):
            pid = int(img_path.split("/")[-1].split("_")[1])
            if pid not in pid2sar.keys():
                continue
            for sar_path in pid2sar[pid]:
                dataset_pair.append([(img_path, self.pid_begin + pid, 0), (sar_path, self.pid_begin + pid, 1)])

        max_pid = max(pid_container)

        return dataset, dataset_pair, max_pid
