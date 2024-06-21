import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from utils import *

from DCNv2_latest.dcn_v2 import DSP_sep2


def feat_warp(x, flow):
    global flag
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device

    grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(x,grid_flow,mode='bilinear',padding_mode='border',align_corners=True)
    return output

class img_warp(torch.nn.Module):
    def __init__(self):
        super(img_warp, self).__init__()

        self.num_heads = 8
        self.heads_dim = 32 // self.num_heads
        self.w_n = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.w_b = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv2d(2*self.heads_dim, 1, kernel_size=3, padding=1, bias=True)
        self.tail = nn.Sequential(
            nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_heads, self.num_heads, kernel_size=3, padding=1, bias=False),
        )
        
    def forward(self, x, flow):
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                             f'flow ({flow.size()[1:3]}) are not the same.')
        
        # create mesh grid
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, device=flow.device, dtype=x.dtype),
                torch.arange(0, W, device=flow.device, dtype=x.dtype))
        grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        grid_flow = grid_flow.type(x.type())
        output_n = F.grid_sample(x, grid_flow, mode='nearest',padding_mode='zeros',align_corners=True)
        output_b = F.grid_sample(x, grid_flow, mode='bilinear',padding_mode='border',align_corners=True)

        output_n = self.w_n(output_n)   
        output_n = output_n.reshape(B, 32, 1, H, W).reshape(B*self.num_heads, self.heads_dim, H, W)  #8B,4,H,W
        output_b = self.w_b(output_b)   
        output_b = output_b.reshape(B, 32, 1, H, W).reshape(B*self.num_heads, self.heads_dim, H, W)  #8B,4,H,W

        multi_out = torch.cat((output_n, output_b), dim=1)
        multi_out = self.conv(multi_out).reshape(B, self.num_heads, H, W)

        multi_out = self.tail(multi_out)
    
        return multi_out

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(31, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )

        self.conv1 = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)

    def forward(self, x_img, x_f):

        x_f = self.head(x_f)
        img = self.conv1(x_f) + x_img
        att = torch.sigmoid(self.conv2(img))
        x_out = self.conv3(x_f) * att
        x_out = self.tail(x_out) + x_f

        return x_out
        
class Re_Weight(torch.nn.Module):
    def __init__(self):
        super(Re_Weight, self).__init__()
        
        self.dim = 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
        )
        self.conv2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
        )

        self.pro1 = Propagate()
        self.pro2 = Propagate()
        
    def forward(self, x, key):

        _, head, _, _ = key.shape

        x_f = self.conv1(x) + x.repeat(1, head, 1, 1)
        key_f = self.conv2(key) + key

        att1 = self.conv3((x_f - key_f)**2)
        key_out = att1 * x_f + key_f
        
        att2 = self.conv4((x_f - key_out)**2)
        x_out = att2 * key_out + x_f

        x_out = self.pro1(x_out)
        key_out = self.pro2(key_out)
    
        return x_out, key_out
    
class Propagate(torch.nn.Module):
    def __init__(self):
        super(Propagate, self).__init__()

        self.dim = 8
        self.head = nn.Conv2d(self.dim, 32, kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(8, 8, kernel_size=1, padding=0, bias=False) 
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True) 
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32*4, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32*4, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )
        self.tail = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):

        x = self.head(x)
        x1 = x[:,0:8,:,:]
        x1 = self.conv1(x1)
        x2 = torch.cat([x1, x[:,8:16,:,:]], dim=1)
        x2 = self.conv2(x2)
        x3 = torch.cat([x2, x[:,16:24,:,:]], dim=1)
        x3 = self.conv3(x3)
        x4 = torch.cat([x3, x[:,24:32,:,:]], dim=1)
        x4 = self.conv4(x4)

        x_out = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        x_out = self.tail(x_out + x) 

        return x_out

class Feat_Align(nn.Module):
    def __init__(self):
        super(Feat_Align, self).__init__()

        self.topk = 3
        self.offset_conv1 = nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.dcn_attention = DSP_sep2(
            32,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=8,
        )
        self.conv = nn.Conv2d(self.topk, 1, 3, 1, 1, bias=True)

        self.tail = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32*4, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32*4, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, key):

        B, C, H, W = x.shape

        offset = torch.cat([x, key], dim=1)  #B, 2*C, H, W
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))  #B, 2*C, H, W

        warp = self.dcn_attention(key, offset).reshape(B, C, 9, H, W)   #(B,Cx9,H,W)
        q = warp.permute(0, 3, 4, 2, 1).reshape(B, H*W, 9, C)
        k = x.reshape(B, C, H*W).permute(0, 2, 1).unsqueeze(3)   #(B,HW,C,1)
        corr = torch.matmul(q, k)
        corr, slt_idx = torch.topk(corr, self.topk, dim=2, largest=True, sorted=True)  #B, H*W, topk, 1

        warp = warp.reshape(B*C, 9, H*W)   
        slt_idx = slt_idx.reshape(B, H*W, self.topk).permute(0, 2, 1)     #B,topk,HW
        key_warp = []
        for i in range(self.topk):
            idx = slt_idx[:, i, :]
            idx = idx.repeat(C, 1).unsqueeze(1)
            warp_temp = torch.gather(warp, 1, idx)
            if i == 0:
                key_warp = warp_temp
            else:
                key_warp = torch.cat((key_warp, warp_temp), dim=1)   #(BC,topk,HW)

        key_warp = key_warp.reshape(B*C, self.topk, H, W)  #BC,topk,H,W
        key_warp = self.conv(key_warp).reshape(B, C, H, W)  # B, C, H, W
        key_warp = self.tail(key_warp)

        return key_warp

class Fusion(torch.nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.res = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32*4, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32*4, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32*4, 32, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        )
        self.tail = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x, key1, key2):

        B, C, H, W = x.shape

        corr_x = x.reshape(B*C, H*W).unsqueeze(2).unsqueeze(3)  #(BC,HW,1,1)
        corr_key1 = key1.reshape(B*C, H*W).unsqueeze(2).unsqueeze(3) #(BC,HW,1,1)
        corr_key2 = key2.reshape(B*C, H*W).unsqueeze(2).unsqueeze(3)

        weight1 = torch.matmul(corr_x, corr_key1).view(B*C, H*W).unsqueeze(0)   #(1,BC,HW)
        weight2 = torch.matmul(corr_x, corr_key2).view(B*C, H*W).unsqueeze(0)
        weight_maps = torch.stack([weight1, weight2], dim=0)   #(2,BC,HW)
        weight_maps = torch.softmax(weight_maps, dim=0).reshape(2, B, C, H, W)  #(2,B,C,H,W)
        weight1 = weight_maps[0].squeeze(1)
        weight2 = weight_maps[1].squeeze(1)

        out = weight1 * key1 + weight2 * key2 + x
        out = self.res(out)

        out = self.tail(out)

        return out

class FuNet(torch.nn.Module):
    def __init__(self):
        super(FuNet, self).__init__()

        self.warp1 = img_warp()
        self.warp2 = img_warp()
        self.reweight1 = Re_Weight()
        self.reweight2 = Re_Weight()
        self.corr1 = SAM()
        self.corr2 = SAM()
        self.corr3 = SAM()
        self.flownet1 = Feat_Align()
        self.flownet2 = Feat_Align()
        self.fusion = Fusion()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
        )
        
    def forward(self, x, key1, key2, flow1, flow2):  #flow(n, 2, h, w)
        #图像变形
        key1_warp = self.warp1(key1[:, :1, :, :], flow1.permute(0, 2, 3, 1))    #input_flow(n, h, w, 2)
        key2_warp = self.warp2(key2[:, :1, :, :], flow2.permute(0, 2, 3, 1))    #B,8,H,W

        #预测帧
        pre_x1, pre_key1 = self.reweight1(x[:, :1, :, :], key1_warp)
        pre_x2, pre_key2 = self.reweight2(x[:, :1, :, :], key2_warp)

        #图像信息传递到特征
        x_img = x[:, :1, :, :]
        x_img = self.conv(torch.cat([pre_x1, x_img, pre_x2], dim=1))
        key1_warp = feat_warp(key1[:, 1:, :, :], flow1.permute(0, 2, 3, 1))
        key2_warp = feat_warp(key2[:, 1:, :, :], flow2.permute(0, 2, 3, 1))

        x_f = self.corr1(x_img, x[:, 1:, :, :])
        key1_f = self.corr2(pre_key1, key1_warp)
        key2_f = self.corr3(pre_key2, key2_warp) 
        
        #结合图像光流进行特征再对齐(可变形卷积)
        key1_warp = self.flownet1(x_f, key1_f)
        key2_warp = self.flownet2(x_f, key2_f)

        #归一融合输出
        out = self.fusion(x_f, key1_warp, key2_warp) + x[:, :1, :, :]

        return out