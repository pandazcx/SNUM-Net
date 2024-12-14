from PIL import Image
import cv2
import time
from importlib import import_module
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
import shutil
from metrics.metrics import ref_evaluate,no_ref_evaluate
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import torch.nn.functional as F
import os
import torch
import numpy as np

from einops import rearrange
import matplotlib.pyplot as plt
sys.path.append("..")
# from utils import *
import Loss.loss as loss


def remove_module_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            k = key.replace("module.", "")
        else:
            k = key
        new_state_dict[k] = state_dict[key]
    return new_state_dict

def make_model(args):
    if args.path:
        config_path = os.path.join(args.path, "config.yml")

    else:
        config_path = "config.yml"

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    network = import_module("network." + config["Version"])
    model = network.DeepMSN_Net(config["network"])
    return model,config


class Runner:
    def __init__(self, model, args, config, rank):

        torch.cuda.set_device(rank)
        self.config = config
        self.rank = rank

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.rank)

        if args.path:
            self.load_checkpoint(args)
        self.model = DDP(self.model)
        if args.recode:
            self.initial_train_recode(args) if args.mode == "train" else self.initial_test_recode(args)

    def make_optimizer(self):
        lr = self.config["train"]["optim"]["init_lr"]
        wd = self.config["train"]["optim"]["weight_decay"]
        bs = self.config["train"]["optim"]["betas"]
        if self.config["train"]["optim"]["type"] == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr = lr, weight_decay = wd,betas = bs)
        else:
            return torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = wd,betas = bs)
    def make_loss(self):
        type = self.config["train"]["loss_type"]
        if type == "mse":
            return nn.MSELoss()
        elif type == "mix1":
            return loss.MIX1Loss()
        elif type == "mix2":
            return loss.MIX2Loss()
        elif type == "mix3":
            return loss.MIX3Loss()#.to(self.rank)
        elif type == "Charbonnier":
            return loss.CharbonnierLoss()
        elif type == "l1":
            return nn.L1Loss()

    def make_scheduler(self,optimizer,current_idx,train_loader):
        type = self.config["train"]["optim"]["scheduler_type"]
        if type == "linear":
            end_factor = self.config["train"]["optim"]["final_lr"] / self.config["train"]["optim"]["init_lr"]
            return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=end_factor,
                                                     total_iters=len(train_loader) * self.config["train"]["epoch"],
                                                     last_epoch=current_idx - 1, verbose=False)
        elif type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * self.config["train"]["epoch"],
                                                              eta_min=self.config["train"]["optim"]["final_lr"],last_epoch = current_idx - 1)


    def initial_test_recode(self,args):
        self.save_path = os.path.join(args.path, ("Test-" + args.sensor + "-" + str(self.checkpoints["current_idx"])))
        if self.rank == 0:
            os.makedirs(self.save_path,exist_ok=True)
            logging.basicConfig(filename=os.path.join(self.save_path, "test_recode.log"),
                                format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            logging.disable(logging.CRITICAL)



    def initial_train_recode(self,args):
        timestr = time.strftime("%Y%m%d-%H%M%S")[4:]

        if args.path and self.rank == 0:
            self.save_path = args.path
            recode_path = os.path.join(self.save_path, ("finetune-" + args.sensor + "-" + timestr))
            os.makedirs(recode_path)
            shutil.copy(os.path.join(self.save_path, "config.yml"),
                        os.path.join(recode_path, "config.yml"))

        if not args.path and self.rank == 0:
            self.save_path = os.path.join("Recode",("recode-" + self.config["Version"] + "-" + args.sensor + "-" + timestr))
            recode_path = os.path.join(self.save_path, "First_time")
            os.makedirs(recode_path)
            shutil.copy("config.yml", os.path.join(self.save_path, "config.yml"))
            shutil.copy("config.yml", os.path.join(recode_path, "config.yml"))

        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=recode_path)
            logging.basicConfig(filename=os.path.join(recode_path, "train_recode.log"),
                                format='%(asctime)s %(message)s',level=logging.INFO)
            logging.info(self.config)

        else:
            self.writer = None
            logging.disable(logging.CRITICAL)

    def load_checkpoint(self, args):
        path = args.path
        recoder = "finetune : {}".format(path)
        self.checkpoints = torch.load(os.path.join(path, self.config["train"]["load"]["model"]))
        if self.rank == 0:
            print(recoder)
            self.model.load_state_dict(remove_module_dict(self.checkpoints["model_state_dict"]))

    def save_checkpoint(self,*args):
        if self.rank == 0:
            checkpoint = {"current_epoch": args[0],
                          "current_idx": args[1],
                          "model_state_dict": args[2],
                          "optimizer_state_dict": args[3]}
            torch.save(checkpoint, args[4])

    def save_img(self,img, folder, name):
            folder = os.path.join(self.save_path, folder)
            os.makedirs(folder, exist_ok=True)
            # cv2.imwrite(os.path.join(folder, name), img[:,:,3])
            # img = Image.fromarray(img, mode='CMYK').convert('RGB')
            # img.save(os.path.join(folder, name))
            cv2.imwrite(os.path.join(folder, name), img)

    def save_diff(self, img, gt, folder, name):
        folder = os.path.join(self.save_path, folder)
        os.makedirs(folder, exist_ok=True)
        diff = np.abs(img - gt)  # 计算绝对差
        diff = np.mean(diff, axis=2)
        print(diff.shape)
        plt.imshow(diff, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Difference')
        plt.title('Heatmap of Mean Difference')
        plt.axis('off')  # 不显示坐标轴
        # 保存图像
        plt.savefig(os.path.join(folder, name), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图像窗口

    def average_loss(self, loss):
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        # loss /= self.world_size
        loss /= dist.get_world_size()
        return loss.item()

    def average_metrice(self, tmp_results):
        all_results = {}
        for key in tmp_results:
            local_data = torch.tensor(tmp_results[key], dtype=torch.float32, device=self.rank)
            all_data = [torch.zeros_like(local_data) for _ in range(dist.get_world_size())]

            dist.all_gather(all_data, local_data)
            gathered_data = torch.cat(all_data)
            global_mean = gathered_data.mean().item()
            all_results[key] = global_mean

        return all_results




    def train(self,args,train_loader,test_reduce_loader,test_full_loader):
        optimizer = self.make_optimizer()
        loss_function = self.make_loss()

        if args.path and self.config["train"]["load"]["inherit"]:
            optimizer.load_state_dict(self.checkpoints["optimizer_state_dict"])
            current_epoch = self.checkpoints["current_epoch"]
            current_idx = self.checkpoints["current_idx"]
        else:
            current_epoch = 0
            current_idx = 0

        scheduler = self.make_scheduler(optimizer,current_idx,train_loader)
        last_epoch = self.config["train"]["epoch"] - current_epoch


        self.model.train()
        for epoch_idx in range(last_epoch):
            epoch = epoch_idx + current_epoch
            total_loss = 0
            total_loss_1 = 0
            total_loss_2 = 0

            train_loader.sampler.set_epoch(epoch)
            batch_list = tqdm(train_loader) if self.rank == 0 else train_loader

            for input_data in batch_list:
                current_lr = round(optimizer.param_groups[0]["lr"], 7)

                output,loss_2 = self.model(input_data["lms"].to(self.rank), input_data["pan"].to(self.rank))
                loss_2 = torch.mean(loss_2)
                optimizer.zero_grad()
                loss_1 = loss_function(output, input_data["gt"].to(self.rank))

                loss = loss_1 + 0.01 * loss_2
                total_loss += loss.data.item()
                total_loss_1 += loss_1.data.item()
                total_loss_2 += loss_2.data.item()

                loss.backward()

                if self.config["train"]["clip_grad"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)
                optimizer.step()
                scheduler.step()
                current_idx += 1

                if self.rank == 0:
                    batch_list.set_description("epoch:%d iter:%d loss:%.4f end_loss:%.4f lr:%.6f"%
                                               (epoch + 1,current_idx, loss.data.item(),loss_1.data.item(),current_lr))

                if (current_idx) % self.config["val"]["RR_freq"] == 0:
                    self.test_reduce(test_reduce_loader)
                if (current_idx) % self.config["val"]["FR_freq"] == 0:
                    self.test_full(test_full_loader,args.sensor)

                if args.recode and self.rank == 0:
                    self.writer.add_scalar('loss', loss.data.item(), current_idx)
                    if current_idx % self.config["save"]["auto_freq"] == 0:
                        name = os.path.join(self.save_path, 'model_current.pth')
                        self.save_checkpoint(epoch, current_idx, self.model.state_dict(), optimizer.state_dict(), name)
                    if current_idx % self.config["save"]["freq"] == 0:
                        name = os.path.join(self.save_path, 'model{}.pth'.format(current_idx))
                        self.save_checkpoint(epoch, current_idx, self.model.state_dict(), optimizer.state_dict(), name)

            if args.recode:
                avg_loss = self.average_loss(torch.tensor(total_loss).cuda(self.rank)) / len(train_loader)
                avg_loss_1 = self.average_loss(torch.tensor(total_loss_1).cuda(self.rank)) / len(train_loader)
                avg_loss_2 = self.average_loss(torch.tensor(total_loss_2).cuda(self.rank)) / len(train_loader)

                logging.info("[idx %d][epoch %d] ave_loss: %.4f ave_loss_1: %.4f ave_loss_2: %.4f learning_rate: %.6f" % (
                        current_idx + 1, epoch + 1, avg_loss, avg_loss_1, avg_loss_2,  current_lr))


    def test_reduce(self,test_loader, save_img = False):
        if self.rank == 0:
            print("======= test_reduce ========")
            logging.info("======= test_reduce ========")

        self.model.eval()
        tmp_results = {'PSNR':[],'SSIM':[],'SAM':[], 'ERGAS':[], 'SCC':[], 'Q':[], 'Time':[]}
        with torch.no_grad():
            idx = 0
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader
            for input_data in batch_list:

                start_time = time.perf_counter()
                output,_ = self.model(input_data["lms"].to(self.rank), input_data["pan"].to(self.rank))
                end_time = time.perf_counter()

                if save_img:
                    output_save = self.to_rgb(output.squeeze(0))
                    output_save = (output_save * 255).astype(np.uint8)
                    self.save_img(img=output_save, folder="Reduce",
                                  name="%03d.png" % (idx * dist.get_world_size() + self.rank))
                    idx += 1

                output = torch.clip(output,0.,1.)

                output = (output[0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
                gt = (input_data["gt"][0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)

                results = ref_evaluate(output,gt)
                tmp_results['Time'].append(end_time - start_time)
                tmp_results['SAM'].append(results["SAM"])
                tmp_results['SCC'].append(results["SCC"])
                tmp_results['Q'].append(results["Q"])
                tmp_results['ERGAS'].append(results["ERGAS"])
                tmp_results['PSNR'].append(results["PSNR"])
                tmp_results['SSIM'].append(results["SSIM"])

                if self.rank == 0:
                    batch_list.set_description("PSNR:%.2f" % results["PSNR"])




        aver_results = self.average_metrice(tmp_results)
        if self.rank == 0:
            for key in aver_results.keys():
                logging.info(f'{key} metric value: {aver_results[key]:.4f}')
                print("\n")
                print(f'{key} metric value: {aver_results[key]:.4f}')



    def test_full(self,test_loader, sensor, save_img = False):
        if self.rank == 0:
            print("======= test_full ========")
            logging.info("======= test_full ========")

        self.model.eval()
        tmp_results = {'D_l': [], 'D_s': [], 'QNR': [], 'Time': []}
        with torch.no_grad():
            idx = 0
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader
            for input_data in batch_list:

                # lms = input_data["ms"].to(self.rank)
                # pan = input_data["pan"].to(self.rank)
                # _, _, h, w = pan.shape
                # lms = F.interpolate(lms, size=[h, w], mode='bicubic', align_corners=True)



                start_time = time.perf_counter()
                # output, _ = self.model(lms, pan)
                output,_ = self.model(input_data["lms"].to(self.rank), input_data["pan"].to(self.rank))
                end_time = time.perf_counter()


                # start_time = time.perf_counter()
                # HQ_up = F.interpolate(HQ, size=(H, W), mode='bicubic',align_corners=True)
                # HQ_up, bls = window_partitionx(HQ_up, 256)
                # FR_up,_ = window_partitionx(FR,256)
                # output = self.model(HQ_up.to(self.device), FR_up.to(self.device))
                # output = window_reversex(output,256,H,W,bls)
                # end_time = time.perf_counter()
                if save_img :
                    output_save = self.to_rgb(output.squeeze(0))
                    output_save = (output_save * 255).astype(np.uint8)
                    self.save_img(img=output_save, folder="Full", name="%03d.png" % (idx * dist.get_world_size() + self.rank)) #.tif
                    idx += 1

                output = torch.clip(output, 0., 1.)

                output = (output[0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
                pan = (input_data["pan"][0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
                ms = (input_data["ms"][0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)
                lms = (input_data["lms"][0].permute(1, 2, 0).cpu().numpy()).astype(np.float32)

                tmp_results['Time'].append(end_time - start_time)
                results = no_ref_evaluate(output,pan,ms,lms,sensor=sensor,flagQNR=1)

                tmp_results['D_l'].append(results["D_lambda"])
                tmp_results['D_s'].append(results["D_s"])
                tmp_results['QNR'].append(results["QNR"])

                if self.rank == 0:
                    batch_list.set_description("D_lambda: %.4f" % results["D_lambda"])



        aver_results = self.average_metrice(tmp_results)
        if self.rank == 0:
            for key in aver_results.keys():
                logging.info(f'{key} metric value: {aver_results[key]:.4f}')
                print("\n")
                print(f'{key} metric value: {aver_results[key]:.4f}')


    def to_rgb(self, x: torch.Tensor | np.ndarray, tol_low=0.01, tol_high=0.99):
        c = x.shape[0]

        if c == 4:
            x = x[[2, 1, 0], :, :]
        elif c == 8:
            x = x[[4, 2, 1], :, :]
        else:
            raise ValueError(f"Unsupported channel number: {c}")
        c, h, w = x.shape
        x = rearrange(x, 'c h w -> c (h w)')
        sorted_x, _ = torch.sort(x, dim=1)
        t_low = sorted_x[:, int(h * w * tol_low)].unsqueeze(1)
        t_high = sorted_x[:, int(h * w * tol_high)].unsqueeze(1)
        x = torch.clamp((x - t_low) / (t_high - t_low), 0, 1)
        x = rearrange(x, 'c (h w) -> h w c',c=c, h=h, w=w)
        return x.cpu().numpy()
