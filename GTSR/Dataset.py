import os
import os.path
import numpy as np
import random
from PIL import Image
import torch
import cv2
import glob
import torch.utils.data as udata
import torch.nn.functional as F

class Dataset(udata.Dataset):
    def __init__(self,win, path,aug_mode,scale=16, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.win = win
        self.scale = scale
        self.aug_mode = aug_mode
        self.HQ_list = []
        self.LQ_list = []
        self.Guide_list = []
        suffix = "train" if train else "val"
        HQ_dir = os.path.join(path,"thermal",suffix, "GT")
        LQ_dir = os.path.join(path,"thermal",suffix, "LR_x%d"%scale)
        Guide_dir = os.path.join(path,"visible",suffix)

        path_tmp = os.listdir(HQ_dir)
        path_tmp = [os.path.join(HQ_dir,s) for s in path_tmp]
        self.HQ_list += path_tmp

        path_tmp = os.listdir(LQ_dir)
        path_tmp = [os.path.join(LQ_dir,s) for s in path_tmp]
        self.LQ_list += path_tmp

        path_tmp = os.listdir(Guide_dir)
        path_tmp = [os.path.join(Guide_dir,s) for s in path_tmp]
        self.Guide_list += path_tmp

        self.HQ_list.sort()
        self.LQ_list.sort()
        self.Guide_list.sort()


    def argument(self, hq,lq,guide):

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        if hflip:
            hq = torch.flip(hq, dims=[2]).clone()
            lq = torch.flip(lq, dims=[2]).clone()
            guide = torch.flip(guide, dims=[2]).clone()
        if vflip:
            hq = torch.flip(hq, dims=[1]).clone()
            lq = torch.flip(lq, dims=[1]).clone()
            guide = torch.flip(guide, dims=[1]).clone()

        return hq, lq, guide

    def get_patch_random(self, hq, lq, guide):
        win = self.win
        h, w = hq.shape[1:]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        hq = hq[:, y:y + win, x:x + win]
        lq = lq[:, y:y + win, x:x + win]
        guide = guide[:, y:y + win, x:x + win]
        return hq,lq,guide


    def load_file(self,idx):

        HQ_data = cv2.imread(self.HQ_list[idx], 0)
        h, w = HQ_data.shape
        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = cv2.imread(self.LQ_list[idx], 0)
        LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))
        LQ_data = np.expand_dims(LQ_data, axis=-1)
        Guide_data = cv2.imread(self.Guide_list[idx],1)
        Guide_data = cv2.cvtColor(Guide_data, cv2.COLOR_BGR2RGB)

        return HQ_data,LQ_data,Guide_data

    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor / 255.
        return img_tensor

    def __len__(self):
        return len(self.HQ_list)

    def __getitem__(self, idx):
        HQ_data,LQ_data,Guide_data = self.load_file(idx)

        HQ = self.totensor(HQ_data)
        LQ = self.totensor(LQ_data)
        Guide = self.totensor(Guide_data)

        if self.train:
            if self.aug_mode:
                HQ, LQ, Guide = self.argument(HQ, LQ, Guide)
            HQ, LQ, Guide = self.get_patch_random(HQ, LQ, Guide)

        return {'HQ': HQ,
                'Guide': Guide,
                'LQ': LQ}


