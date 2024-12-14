import cv2
import os
import os.path
import numpy as np
import torch
import torch.utils.data as udata
import h5py


class Dataset(udata.Dataset):
    def __init__(self,path,img_scale,train):
        super(Dataset, self).__init__()
        self.img_scale = img_scale
        if train:
            for set_path in path:
                data = h5py.File(set_path)
                self.process_ms(data)
                self.process_lms(data)
                self.process_pan(data)
                self.process_gt(data)
        else:
            data = h5py.File(path)
            self.process_ms(data)
            self.process_lms(data)
            self.process_pan(data)
            self.process_gt(data)

    def process_gt(self, data):
        if data.get('gt', None) is None:
            self.gt = self.lms
        else:
            gt = data["gt"][...]  # convert to np tpye for CV2.filter
            gt = np.array(gt, dtype=np.float32) / self.img_scale
            self.gt = torch.from_numpy(gt)  # NxCxHxW:

    def process_lms(self, data):
        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        lms = np.array(lms, dtype=np.float32) / self.img_scale
        self.lms = torch.from_numpy(lms)

    def process_ms(self, data):
        ms = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms = np.array(ms, dtype=np.float32) / self.img_scale
        self.ms = torch.from_numpy(ms) # NxCxHxW:

    def process_pan(self, data):
        pan = data['pan'][...]  # Nx1xHxW
        pan = np.array(pan, dtype=np.float32) / self.img_scale
        self.pan = torch.from_numpy(pan)




    def __getitem__(self, index):
        return {'gt': self.gt[index, :, :, :],
                'lms': self.lms[index, :, :, :],
                'ms': self.ms[index, :, :, :],
                'pan': self.pan[index, ...]}

    def __len__(self):
        return self.gt.shape[0]
