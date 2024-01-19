import torch
import numpy as np
import cv2

def to_ima(out, row, col):
    x_output = out.squeeze(0).squeeze(0)
    Prediction_value = x_output.cpu().data.numpy()
    X_rec = np.clip(Prediction_value[:row, :col], 0, 1)*255
    return X_rec

def graytensor_to_RGB(input):  # (n,1,H,W)
    input = input.repeat(1, 3, 1, 1)
    output = torch.clamp(input, 0, 1)

    return output

def imread_CS_py(Iorg):
    block_size = 32
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.math.log10(PIXEL_MAX / np.math.sqrt(mse))

def PSNR(img1, img2):
    img1 = torch.squeeze(img1)
    img2 = torch.squeeze(img2)
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    img1 = np.clip(img1, 0, 1)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    max_gray = 1.
    mse = np.mean(np.power(img1 - img2, 2))

    return 10. * np.log10(max_gray ** 2 / mse)


def load_ima(imgName, device):
    Img = cv2.imread(imgName, 1)
    #Img = Img.astype(np.float32)
    #Img = Img[200:520, 480:800]
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Img_rec_yuv = Img_yuv.copy()

    Iorg_y = Img_yuv[:, :, 0]
    [Iorg, row, col, Ipad, _, _] = imread_CS_py(Iorg_y)
    Img_output = Ipad / 255.

    batch_x = torch.from_numpy(Img_output)
    batch_x = batch_x.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    batch_x = batch_x.unsqueeze(0).unsqueeze(0)

    return batch_x, row, col, Iorg, Img_rec_yuv

def load_crop_ima(imgName, device):
    Img = cv2.imread(imgName, 1)
    Img = Img[200:520, 480:800]  #中心区域320
    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
    Img_rec_yuv = Img_yuv.copy()

    Iorg_y = Img_yuv[:, :, 0]
    [Iorg, row, col, Ipad, _, _] = imread_CS_py(Iorg_y)
    Img_output = Ipad / 255.

    batch_x = torch.from_numpy(Img_output)
    batch_x = batch_x.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    batch_x = batch_x.unsqueeze(0).unsqueeze(0)

    return batch_x, row, col, Iorg, Img_rec_yuv


def loss_fn(img1,img2):
    def charbonnier_loss(img1,img2):
        diff = img1 - img2
        eps = 1e-3
        loss = torch.mean(torch.sqrt((diff * diff) + (eps * eps)))
        return loss
    loss1 = charbonnier_loss(img1,img2)
    # img1_fft = torch.fft.fft2(img1, dim=(-2, -1))
    # img1_fft = torch.stack((img1_fft.real, img1_fft.imag), -1)
    # img2_fft = torch.fft.fft2(img2, dim=(-2, -1))
    # img2_fft = torch.stack((img2_fft.real, img2_fft.imag), -1)
    # loss2 = charbonnier_loss(img1_fft, img2_fft)
    # lam = 0.05
    return loss1

def count_parameters(model):   
    num_params = 0
    for para in model.parameters():
        num_params += para.numel()
#         print('Layer %d' % num_count)
#         print(para.size())
    print("para num: %d = %.04f M " % (num_params, num_params//1000000))