import os
import os.path
import numpy as np
import random
import h5py
import torch
# import cv2
from PIL import Image
import glob
import torch.utils.data as udata
import torch.nn.functional as F
# from torchvision import transforms


class Dataset_NYU_past(udata.Dataset):
    def __init__(self,win,path,aug_mode,scale,train = True):
        super(Dataset_NYU_past, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode
        self.scale = scale
        HQ_dir = os.path.join(path, "depth.npy")
        Guide_dir = os.path.join(path, "rgb.npy")
        self.HQ_list = np.load(HQ_dir)
        self.Guide_list = np.load(Guide_dir)
        if train:
            self.minmax_list = None
        else:
            minmax_dir = os.path.join(path, "NYU_test_minmax.npy")
            self.minmax_list = np.load(minmax_dir)


    def argument(self, hq,guide, mode):
        if mode == 1:
            hq_aug = hq
            guide_aug = guide
        elif mode == 2:
            hq_aug = np.flip(hq,[0]).copy()
            guide_aug = np.flip(guide,[0]).copy()
        elif mode == 3:
            hq_aug = np.flip(hq,[1]).copy()
            guide_aug = np.flip(guide,[1]).copy()
        elif mode == 4:
            hq_aug = np.flip(hq,[0,1]).copy()
            guide_aug = np.flip(guide,[0,1]).copy()

        return hq_aug, guide_aug

    def get_patch_random(self, hq, guide):
        win = self.win
        h, w = hq.shape[:2]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        hq = hq[y:y + win, x:x + win]
        guide = guide[y:y + win, x:x + win,:]
        return hq,guide



    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor

    def __len__(self):
        return self.Guide_list.shape[0]

    def __getitem__(self, idx):

        Guide_data = self.Guide_list[idx]
        HQ_data = self.HQ_list[idx]

        if self.train:
            HQ_data, Guide_data = self.argument(HQ_data, Guide_data, np.random.randint(1, self.aug_mode))
            HQ_data, Guide_data = self.get_patch_random(HQ_data, Guide_data)
            minmax_data = torch.tensor([0,0])
        else:
            minmax_data = torch.from_numpy(self.minmax_list[:, idx])

        h, w = HQ_data.shape[:2]
        LQ_data = np.array(Image.fromarray(HQ_data).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))



        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = np.expand_dims(LQ_data, axis=-1)


        return {'HQ': self.totensor(HQ_data),
                'Guide': self.totensor(Guide_data),
                'LQ': self.totensor(LQ_data),
                'minmax': minmax_data}


class Dataset_Middle_LU(udata.Dataset):
    def __init__(self,path,scale):
        super(Dataset_Middle_LU, self).__init__()
        self.scale = scale
        HQ_dir = os.path.join(path, "depth")
        Guide_dir = os.path.join(path, "rgb")

        files = os.listdir(HQ_dir)
        files.sort()
        self.HQ_list = [os.path.join(HQ_dir, file) for file in files]

        files = os.listdir(Guide_dir)
        files.sort()
        self.Guide_list = [os.path.join(Guide_dir, file) for file in files]


    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor

    def modcrop(self,img):
        h, w = img.shape[0], img.shape[1]
        h = h - h % self.scale
        w = w - w % self.scale
        return img[:h, :w]

    def load_depth(self, depth_data):
        depth_data = (np.array(Image.open(depth_data)) / 255.).astype(np.float32)
        return self.modcrop(depth_data)

    def load_color(self, color_data):
        color_data = (np.array(Image.open(color_data)) / 255. ).astype(np.float32)
        return self.modcrop(color_data)



    def __len__(self):
        return len(self.Guide_list)

    def __getitem__(self, idx):

        Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_depth(self.HQ_list[idx])


        h, w = HQ_data.shape[:2]
        LQ_data = np.array(Image.fromarray(HQ_data).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))

        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = np.expand_dims(LQ_data, axis=-1)


        return {'HQ': self.totensor(HQ_data),
                'Guide': self.totensor(Guide_data),
                'LQ': self.totensor(LQ_data)}


class Dataset_RGBDD(udata.Dataset):
    def __init__(self, path, scale, syn):
        super(Dataset_RGBDD, self).__init__()
        self.scale = scale
        self.syn = syn
        types = ['models', 'plants', 'portraits']
        self.HQ_list = []
        self.Guide_list = []
        self.LQ_list = []
        for type in types:
            list_dir = os.listdir('%s/%s/%s_test' % (path, type, type))
            for n in list_dir:
                self.Guide_list.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (path, type, type, n, n))
                self.HQ_list.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (path, type, type, n, n))
                self.LQ_list.append('%s/%s/%s_test/%s/%s_LR_fill_depth.png' % (path, type, type, n, n))

    def totensor(self, img):
        img = np.ascontiguousarray(img)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
        return img_tensor

    def load_depth(self, depth_data):
        return np.array(Image.open(depth_data)).astype(np.float32)

    def load_color(self, color_data):
        return np.array(Image.open(color_data).convert("RGB")).astype(np.float32)

    def __len__(self):
        return len(self.Guide_list)

    def __getitem__(self, idx):
        Guide_data = self.load_color(self.Guide_list[idx])
        HQ_data = self.load_depth(self.HQ_list[idx])

        h, w = HQ_data.shape[:2]
        if self.syn:
            LQ_data = np.array(Image.fromarray(HQ_data).resize((w // self.scale, h // self.scale), Image.BICUBIC))
        else:
            LQ_data = np.array(Image.open(self.LQ_list[idx]).resize((w // self.scale, h // self.scale), Image.BICUBIC)).astype(np.float32)

        maxx = np.max(LQ_data)
        minn = np.min(LQ_data)
        LQ_data = (LQ_data - minn) / (maxx - minn)

        guide_max = np.max(Guide_data)
        guide_min = np.min(Guide_data)
        Guide_data = (Guide_data - guide_min) / (guide_max - guide_min)


        LQ_data = np.array(Image.fromarray(LQ_data).resize((w, h), Image.BICUBIC))

        HQ_data = np.expand_dims(HQ_data, axis=-1)
        LQ_data = np.expand_dims(LQ_data, axis=-1)
        minmax = torch.tensor([maxx,minn])

        return {'HQ': self.totensor(HQ_data),
                'Guide': self.totensor(Guide_data),
                'LQ': self.totensor(LQ_data),
                'minmax': minmax}

class Dataset_NYU(udata.Dataset):
    def __init__(self,win,path,aug_mode,scale,train = True):
        super(Dataset_NYU, self).__init__()
        self.train = train
        self.win = win
        self.aug_mode = aug_mode
        self.scale = scale

        HQ_dir = os.path.join(path, "depth.npy")
        Guide_dir = os.path.join(path, "rgb.npy")
        LQ_dir = os.path.join(path, "depth_x%d.npy"%scale)

        self.HQ_list = self.totensor(np.expand_dims(np.load(HQ_dir), axis=-1))
        self.Guide_list = self.totensor(np.load(Guide_dir))
        self.LQ_list = self.totensor(np.expand_dims(np.load(LQ_dir), axis=-1))

        if not train:
            minmax_dir = os.path.join(path, "NYU_test_minmax.npy")
            self.minmax_list = torch.from_numpy(np.load(minmax_dir))

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
        h, w = hq.shape[1:3]
        x = random.randrange(0, w - win + 1)
        y = random.randrange(0, h - win + 1)
        hq = hq[:,y:y + win, x:x + win]
        lq = lq[:,y:y + win, x:x + win]
        guide = guide[:,y:y + win, x:x + win]
        return hq,lq,guide

    def totensor(self, data):
        data = np.ascontiguousarray(data)
        data = data.transpose(0, 3, 1, 2)
        data_tensor = torch.from_numpy(data).float()
        return data_tensor

    def __len__(self):
        return self.Guide_list.shape[0]

    def __getitem__(self, idx):
        Guide_data = self.Guide_list[idx]
        HQ_data = self.HQ_list[idx]
        LQ_data = self.LQ_list[idx]

        if self.train:
            if self.aug_mode:
                HQ_data,LQ_data, Guide_data = self.argument(HQ_data, LQ_data,Guide_data)
            HQ_data,LQ_data, Guide_data = self.get_patch_random(HQ_data, LQ_data, Guide_data)
            minmax_data = torch.tensor([0,0])
        else:
            minmax_data = self.minmax_list[:, idx]

        return {'HQ': HQ_data,
                'Guide': Guide_data,
                'LQ': LQ_data,
                'minmax': minmax_data}