import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import math

class SFB(nn.Module):
    def __init__(self, dim,ratio):
        super(SFB, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // ratio, kernel_size=1, stride=1, padding="same", groups=1, bias=False)
        self.conv2 = nn.Conv2d(dim // ratio, dim * 2, kernel_size=1, stride=1, padding="same", groups=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x,y):
        x_pool = self.pool(x)
        y_pool = self.pool(y)
        att = x_pool * y_pool

        att = self.relu(self.conv1(att))
        x_pool,y_pool = self.sigmoid(self.conv2(att)).chunk(2,dim=1)
        return x * x_pool + y * y_pool

class CB(nn.Module):
    def __init__(self, dim, ratio = 1, ks = 3):
        super(CB, self).__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(dim, dim * ratio, kernel_size=ks, stride=1, padding="same", groups=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim * ratio, dim * ratio, kernel_size=ks, stride=1, padding="same", groups=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim * ratio, dim, kernel_size=ks, stride=1, padding="same", groups=1, bias=False))
    def forward(self,x):
        return self.cb(x) + x

class HEB(nn.Module):
    def __init__(self, dim, ratio_spatial = 1, window_size = 32, sigma = 8):
        super(HEB, self).__init__()
        self.dim = dim
        self.scb1 = CB(dim, ratio_spatial, ks=3)
        self.scb2 = CB(dim, ratio_spatial, ks=3)
        self.kernel = self.gaussian_kernel(window_size, sigma).repeat(dim, 1, 1, 1)
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, padding="same", groups=1, bias=False)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=1, padding="same", groups=1, bias=False)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, padding="same", groups=1, bias=False)

    @staticmethod
    def gaussian_kernel(size, sigma):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= torch.sum(kernel_2d)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        x_local,x_global = self.conv1(x).chunk(2, dim=1)
        x_local = self.scb1(x_local)
        x_global = x_global + self.CPE(x_global)
        x_global = F.conv2d(x_global,self.kernel.to(x.device), padding="same", groups=self.dim)
        x_global = self.scb2(x_global)
        x = x + self.conv2(torch.cat([x_local,x_global], dim=1))
        return x

class CGRU(nn.Module):
    def __init__(self,dim, ks = 3):
        super(CGRU,self).__init__()
        self.sigmiod = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_z = nn.Conv2d(dim * 2, dim, kernel_size=ks, stride=1, padding="same", groups=1, bias=True)
        self.conv_r = nn.Conv2d(dim * 2, dim, kernel_size=ks, stride=1, padding="same", groups=1, bias=True)
        self.conv_h = nn.Conv2d(dim * 2, dim, kernel_size=ks, stride=1, padding="same", groups=1, bias=True)

    def forward(self,h,x):
        z = self.conv_z(torch.cat((h,x),dim=1))
        z = self.sigmiod(z)
        r = self.conv_r(torch.cat((h,x),dim=1))
        r = self.sigmiod(r)
        hn = self.conv_h(torch.cat((r * h, x),dim=1))
        hn = self.tanh(hn)
        h = h * z + hn * (1 - z)
        return h
class UNet(nn.Module):
    def __init__(self, in_dim, feature_dim, ratio_spatial, ratio_fusion, middle_blk_num=2, enc_blk_nums=[1,1], dec_blk_nums=[1,1]):
        super(UNet,self).__init__()
        self.to_hidden = nn.Conv2d(in_dim, feature_dim, kernel_size=3, stride=1, padding="same", groups=1, bias=False)
        self.project_out = nn.Conv2d(feature_dim, in_dim, kernel_size=3, stride=1, padding="same", groups=1, bias=False)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fusions = nn.ModuleList()

        dim = feature_dim
        self.encoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(enc_blk_nums[0])]))
        self.downs.append(nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, bias=False,padding="same"), nn.PixelUnshuffle(2)))

        dim = dim * 2
        self.encoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(enc_blk_nums[1])]))
        self.downs.append(nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, bias=False,padding="same"), nn.PixelUnshuffle(2)))

        dim = dim * 2
        self.middle_blks = nn.Sequential(*[HEB(dim,ratio_spatial) for _ in range(middle_blk_num)])


        self.ups.append(nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, bias=False,padding="same"), nn.PixelShuffle(2)))
        dim = dim // 2
        self.fusions.append(SFB(dim,ratio_fusion))
        self.decoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(dec_blk_nums[0])]))

        self.ups.append(nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, bias=False,padding="same"), nn.PixelShuffle(2)))
        dim = dim // 2
        self.fusions.append(SFB(dim, ratio_fusion))
        self.decoders.append(nn.Sequential(*[CB(dim, ratio_spatial) for _ in range(dec_blk_nums[1])]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        # return inp
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.to_hidden(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, fusion, enc_skip in zip(self.decoders, self.ups, self.fusions, encs[::-1]):
            x = up(x)
            x = fusion(x,enc_skip)
            x = decoder(x)
        x = self.project_out(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class CDB(nn.Module):
    def __init__(self, in_dim, kernel_size = 3):
        super(CDB, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_maker = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, bias=False,padding="same"),
                                          nn.LeakyReLU(),
                                          nn.Conv2d(in_dim, in_dim, kernel_size=3, bias=False,padding="same"),
                                          nn.Conv2d(in_dim, in_dim * kernel_size * kernel_size, kernel_size=1, bias=False,padding="same"))
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self,x,y):
        b, c, h, w = x.shape
        kernel = self.kernel_maker(x + y)
        kernel = kernel.reshape([b, c, self.kernel_size * self.kernel_size, h, w])
        x_unfold = self.unfold(x).reshape(b, c, -1, h, w)
        x = (x_unfold * kernel).sum(2) + x
        return x

class ST(nn.Module):
    def __init__(self, control_nums):
        super(ST, self).__init__()
        self.weight_a = nn.Parameter(torch.randn(control_nums))
        self.weight_b = nn.Parameter(torch.randn(control_nums))
        self.control_nums = control_nums

    def forward(self,x):
        freq = torch.arange(1, self.control_nums + 1, device=x.device) * math.pi
        x = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.control_nums)
        Fourier_sin = torch.sin(freq * x)
        Fourier_cos = torch.cos(freq * x)
        x = torch.sum(self.weight_a * Fourier_sin + self.weight_b * Fourier_cos, dim=-1)
        return x

class Image_update(nn.Module):
    def __init__(self, in_dim,y_dim, feature_dim, ratio_spatial, ratio_fusion, control_nums, middle_blk_num=2, enc_blk_nums=[1,1], dec_blk_nums=[1,1],
                 u_transform=True, training = True):
        super(Image_update, self).__init__()
        self.u_transform = u_transform
        self.matrix_D = CB(in_dim, ratio_spatial)
        self.matrix_Dt = CB(in_dim, ratio_spatial)

        self.matrix_H = nn.Sequential(nn.Conv2d(in_dim, y_dim, kernel_size=1, bias=False, padding="same"), CB(y_dim, ratio_spatial), CB(y_dim, ratio_spatial))
        self.matrix_Ht = nn.Sequential(nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same"), CB(in_dim, ratio_spatial), CB(in_dim, ratio_spatial))
        if u_transform:
            self.matrix_Ht = nn.Sequential(nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same"),
                                           CB(in_dim, ratio_spatial), CB(in_dim, ratio_spatial))
            self.matrix_H = nn.Sequential(nn.Conv2d(in_dim, y_dim, kernel_size=1, bias=False, padding="same"),
                                          CB(y_dim, ratio_spatial), CB(y_dim, ratio_spatial))
        else:
            self.matrix_Ht = nn.Sequential(CB(in_dim, ratio_spatial), CB(in_dim, ratio_spatial))
            self.matrix_H = nn.Sequential(nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same"),
                                          CB(in_dim, ratio_spatial))
        self.s_phi = nn.Sequential(nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, bias=False, padding="same"),
                                   ST(control_nums),
                                   nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, bias=False, padding="same"))
        self.s_psi = nn.Sequential(nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, bias=False, padding="same"),
                                   ST(control_nums),
                                   nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, bias=False, padding="same"))
        self.conv = nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same")
        self.CDB = CDB(in_dim)
        self.solver = UNet(in_dim, feature_dim, ratio_spatial=1, ratio_fusion=ratio_fusion,
                             middle_blk_num=middle_blk_num,
                             enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        self.updater = CGRU(in_dim)
        self.training = training

    def forward(self,u,z,y,l1,l2):
        l1_n = u - self.s_phi(l1 + u) + l1
        l2_n = u - self.s_psi(self.CDB((l2 + u),self.conv(y))) + l2
        z_n = self.matrix_Dt(self.matrix_D(u)-z)
        y_n = self.matrix_Ht(self.matrix_H(u)-y) if self.u_transform else self.matrix_Ht(u-self.matrix_H(y))
        delta = z_n + y_n + l1_n + l2_n
        delta = self.solver(delta)
        u = self.updater(u,delta)
        if self.training:
            l1_n = u - self.s_phi(l1 + u) + l1
            l2_n = u - self.s_psi(self.CDB((l2 + u),self.conv(y))) + l2
            z_n = self.matrix_Dt(self.matrix_D(u)-z)
            y_n = self.matrix_Ht(self.matrix_H(u)-y) if self.u_transform else self.matrix_Ht(u-self.matrix_H(y))
            fu_loss = z_n + y_n + l1_n + l2_n
            fu_loss = F.l1_loss(fu_loss, torch.zeros_like(fu_loss).to(fu_loss.device), reduction='mean')
            return u, fu_loss
        return u, torch.tensor(0.0).to(u.device)

class Lambda_update(nn.Module):
    def __init__(self, in_dim,y_dim,control_nums):
        super(Lambda_update, self).__init__()
        self.s_phi = nn.Sequential(nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, bias=False, padding="same"),
                                   ST(control_nums),
                                   nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, bias=False, padding="same"))
        self.s_psi = nn.Sequential(nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, bias=False, padding="same"),
                                   ST(control_nums),
                                   nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, bias=False, padding="same"))
        self.conv = nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same")
        self.CDB = CDB(in_dim)
        self.updater_1 = CGRU(in_dim)
        self.updater_2 = CGRU(in_dim)

    def forward(self,u,y,l1,l2):
        delta = u - self.s_phi(l1 + u)
        l1 = self.updater_1(l1, delta)
        delta = u - self.s_psi(self.CDB((l2 + u),self.conv(y)))
        l2 = self.updater_2(l2, delta)
        return l1, l2


class SNUM_Net(nn.Module):
    def __init__(self, args, training = True):
        super(SNUM_Net, self).__init__()
        self.ratio_spatial = args["ratio_spatial"]
        self.ratio_fusion = args["ratio_fusion"]
        self.control_nums = args["control_nums"]
        self.middle_blk_num = args["middle_blk_num"]
        self.enc_blk_nums = args["enc_blk_nums"]
        self.dec_blk_nums = args["dec_blk_nums"]
        self.u_transform = args["u_transform"]
        self.training = training

        self.depth = args["depth"]
        self.in_dim = args["in_dim"]
        self.y_dim = args["y_dim"]
        self.feature_dim = args["feature_dim"]


        self.image_updaters = nn.ModuleList(nn.Sequential(*[Image_update(self.in_dim, self.y_dim, self.feature_dim, self.ratio_spatial,
                                                                         self.ratio_fusion, self.control_nums, self.middle_blk_num, self.enc_blk_nums,
                                                                         self.dec_blk_nums,self.u_transform, self.training) for _ in range(self.depth)]))
        self.lambda_updaters = nn.ModuleList(nn.Sequential(*[Lambda_update(self.in_dim, self.y_dim,self.control_nums) for _ in range(self.depth - 1)]))
        self.init = nn.Conv2d(self.in_dim, self.in_dim * 2, kernel_size=3, bias=False, padding="same")

    def forward(self, z,y):
        u = z
        l1,l2 = self.init(u).chunk(2,dim=1)
        fu_ls = torch.zeros(len(self.image_updaters)).to(u.device)
        idx = 0
        for image_updater, lambda_updater in zip(self.image_updaters, self.lambda_updaters):
            u,fu = image_updater(u,z,y,l1,l2)
            fu_ls[idx] = fu
            l1, l2 = lambda_updater(u,y,l1,l2)
            idx += 1
        u,fu = self.image_updaters[-1](u,z,y,l1,l2)
        fu_ls[idx] = fu
        u = u + z
        return u,fu_ls





if __name__ == '__main__':
    import yaml
    from fvcore.nn import parameter_count_table
    from ptflops import get_model_complexity_info

    config_path = "config.yml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = config["network"]

    net = SNUM_Net(args)
    net = torch.nn.DataParallel(net.to("cuda"))
    torch.save(net.state_dict(), "t2_unity_ori.pth")
    # help(get_model_complexity_info)
    inpz = torch.randn(1,4,256,256)
    inpy = torch.randn(1,1,256,256)

    opu,ls = net(inpz,inpy)
    # print(ls)
    # ls = torch.mean(ls)
    # print(ls)
    print(parameter_count_table(net))
    # macs, params = get_model_complexity_info(net, (1,64,64), as_strings=True, print_per_layer_stat=True,
    #                                           verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
