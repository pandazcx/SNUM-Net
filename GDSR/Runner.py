import cv2
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
import shutil
from util import padding,inv_padding,NYU_rmse,calculate_ssim,midd_calc_rmse,rgbdd_calc_rmse
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
sys.path.append("..")
import Loss.loss as loss
import network.model as Model


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
    Train = True if args.mode == "train" else False
    model = Model.SNUM_Net(config["network"], Train)
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
        self.save_path = os.path.join(args.path, ("Test-" + str(self.checkpoints["current_idx"])))
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
            recode_path = os.path.join(self.save_path, ("finetune-" + timestr))
            os.makedirs(recode_path)
            shutil.copy(os.path.join(self.save_path, "config.yml"),
                        os.path.join(recode_path, "config.yml"))

        if not args.path and self.rank == 0:
            self.save_path = os.path.join("Recode",("recode-" + self.config["Version"] + "-"  + timestr))
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




    def train(self,args,train_loader,test_loader_dict):
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

                output,loss_2 = self.model(input_data["LQ"].to(self.rank), input_data["Guide"].to(self.rank))
                loss_2 = torch.mean(loss_2)
                optimizer.zero_grad()
                loss_1 = loss_function(output, input_data["HQ"].to(self.rank))

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

                if (current_idx) % self.config["val"]["freq"] == 0:
                    self.test_NYU(test_loader_dict["NYU"])
                    self.test_Middle_Lu(test_loader_dict["MiddleBury"],"MiddleBury")
                    self.test_Middle_Lu(test_loader_dict["Lu"], "Lu")
                    self.test_RGBDD(test_loader_dict["RGBDD"],True)


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


    def test_iter(self,input_data):
        LQ, pad = padding(8, input_data["LQ"])
        Guide, _ = padding(8, input_data["Guide"])

        start_time = time.perf_counter()
        output,_ = self.model(LQ.to(self.rank), Guide.to(self.rank))
        end_time = time.perf_counter()
        output = inv_padding(pad,output)
        cal_time = end_time - start_time
        return output,cal_time


    def test_NYU(self,test_loader, save_img = False):
        if self.rank == 0:
            print("======= test_NYU ========")
            logging.info("======= test_NYU ========")

        self.model.eval()
        tmp_results = {'RMSE':[],'SSIM':[], 'Time':[]}
        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader
            for batch_idx, input_data in enumerate(batch_list, 0):
                output,cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)

                if save_img:
                    self.save_img(img=np.uint8(output[0,0].cpu().numpy() * 255), folder="NYU",
                                  name="%03d.png" % (batch_idx * dist.get_world_size() + self.rank))

                test_minmax = input_data["minmax"].squeeze(0).to(self.rank)

                rmse = NYU_rmse(HQ[0, 0], output[0, 0], test_minmax)
                ssim = calculate_ssim(HQ, output)

                tmp_results['Time'].append(cal_time)
                tmp_results['RMSE'].append(rmse)
                tmp_results['SSIM'].append(ssim)

                if self.rank == 0:
                    batch_list.set_description("RMSE:%.2f" % rmse)


        aver_results = self.average_metrice(tmp_results)
        if self.rank == 0:
            for key in aver_results.keys():
                logging.info(f'{key} metric value: {aver_results[key]:.4f}')
                print("\n")
                print(f'{key} metric value: {aver_results[key]:.4f}')



    def test_Middle_Lu(self,test_loader, testset, save_img = False):
        if self.rank == 0:
            print("======= test_%s ========"%testset)
            logging.info("======= test_%s ========"%testset)

        self.model.eval()
        tmp_results = {'RMSE': [], 'Time': []}
        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader
            for batch_idx, input_data in enumerate(batch_list, 0):
                output, cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)

                if save_img:
                    self.save_img(img=np.uint8(output[0,0].cpu().numpy() * 255), folder=testset,
                                  name="%03d.png" % (batch_idx * dist.get_world_size() + self.rank))


                rmse = midd_calc_rmse(HQ[0, 0], output[0, 0])

                tmp_results['Time'].append(cal_time)
                tmp_results['RMSE'].append(rmse)

                if self.rank == 0:
                    batch_list.set_description("RMSE:%.2f" % rmse)

        aver_results = self.average_metrice(tmp_results)
        if self.rank == 0:
            for key in aver_results.keys():
                logging.info(f'{key} metric value: {aver_results[key]:.4f}')
                print("\n")
                print(f'{key} metric value: {aver_results[key]:.4f}')


    def test_RGBDD(self,test_loader, is_syn, save_img = False):

        testset = "Syn" if is_syn else "Real"
        if self.rank == 0:
            print("======= test_RGBDD_%s ========"%testset)
            logging.info("======= test_RGBDD_%s ========"%testset)

        self.model.eval()
        tmp_results = {'RMSE': [], 'Time': []}
        with torch.no_grad():
            batch_list = tqdm(test_loader) if self.rank == 0 else test_loader
            for batch_idx, input_data in enumerate(batch_list, 0):
                output,cal_time = self.test_iter(input_data)
                HQ = input_data["HQ"].to(self.rank)

                if save_img:
                    self.save_img(img=np.uint8(output[0,0].cpu().numpy() * 255), folder="RGBDD_%s"%testset,
                                  name="%03d.png" % (batch_idx * dist.get_world_size() + self.rank))

                test_minmax = input_data["minmax"].squeeze(0).to(self.rank)

                rmse = rgbdd_calc_rmse(HQ[0, 0], output[0, 0],test_minmax)

                tmp_results['Time'].append(cal_time)
                tmp_results['RMSE'].append(rmse)

                if self.rank == 0:
                    batch_list.set_description("RMSE:%.2f" % rmse)

        aver_results = self.average_metrice(tmp_results)
        if self.rank == 0:
            for key in aver_results.keys():
                logging.info(f'{key} metric value: {aver_results[key]:.4f}')
                print("\n")
                print(f'{key} metric value: {aver_results[key]:.4f}')

