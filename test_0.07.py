import argparse
import os
import warnings

from utils import *
from Model.SPYNet import SPyNet
from Model.OCTUF import OCT
from Model.CollabNet import FuNet
import glob

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def empty_cache():
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_key = OCT(args.layer_num, args.sr).to(device)
    model_nonkey = OCT(args.layer_num, args.sr).to(device)
    model_flow = FuNet().to(device)
    spynet = SPyNet().to(device)
    
    model_dic = {'model_key': model_key, 'model_nonkey': model_nonkey, 'spynet': spynet, 'model_flow': model_flow}

    spynet.load_state_dict(torch.load('./CheckPoint/spynet.pth'), strict=False)
    pth1_dir = "./CheckPoint/OCT_007.pth"
    checkpoint = torch.load(pth1_dir)
    model_key.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    model_nonkey.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    pth2_dir = "./CheckPoint/CollabNet_007.pth"
    checkpoint = torch.load(pth2_dir)
    model_flow.load_state_dict(checkpoint['net_flow'])

    ext = {'/*.jpg', '/*.png', '/*.tif'}
    video_dir = "./DataSet/REDS/REDS4/"  # the dir of test dataset
    video_dir = glob.glob(video_dir + '/*')
    filepaths = [[] for _ in range(len(video_dir))]
    filepaths_key = [[] for _ in range(len(video_dir))]

    max_filepaths = 0
    for i, cur_video_dir in enumerate(video_dir):
        for img_type in ext:
            filepaths[i] = filepaths[i] + glob.glob(cur_video_dir + img_type)
        filepaths[i].sort()
        max_filepaths = max(max_filepaths, len(filepaths[i]))
        for j, imgName in enumerate(filepaths[i]):
            if j % 8 == 0:
                filepaths_key[i].append(imgName)
            if j == args.eva_len:break
    PSNR_All = np.zeros([max_filepaths, len(video_dir)], dtype=np.float32)
    SSIM_All = np.zeros([max_filepaths, len(video_dir)], dtype=np.float32)
    result_ave = np.zeros([2, len(video_dir) + 1], dtype=np.float32)

    print("Start Reconstruction")

    with torch.no_grad():
        for i in model_dic:
            model_dic[i].eval()
        for i, cur_video_dir in enumerate(video_dir):
            for j, imgName in enumerate(filepaths[i]):
                if j % 8 == 0:
                    F_0, row, col, Iorg, _ = load_ima(imgName, device)
                    out_F_0 = model_dic['model_key'](F_0)
                    x_output = out_F_0[:, :1, :, :].squeeze(0).squeeze(0)
                    Prediction_value = x_output.cpu().data.numpy()
                    X_rec = np.clip(Prediction_value[:row, :col], 0, 1)
                    rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
                    rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
                    PSNR_All[j, i] = rec_PSNR
                    SSIM_All[j, i] = rec_SSIM

                else:
                    frame0 = load_ima(filepaths_key[i][j // 8], device)
                    F_0 = frame0[0]
                    F_1, row, col, Iorg, _ = load_ima(imgName, device)
                    if j // 8 + 1 >= len(filepaths_key):
                        frame8 = load_ima(filepaths_key[i][j // 8], device)
                    else:
                        frame8 = load_ima(filepaths_key[i][j // 8 + 1], device)
                    F_8 = frame8[0]
                    
                    out_F_0 = model_dic['model_key'](F_0)
                    out_F_8 = model_dic['model_key'](F_8)
                    init_F_1 = model_dic['model_nonkey'](F_1)
                    flow_input0 = graytensor_to_RGB(out_F_0[:, :1, :, :])
                    flow_input1 = graytensor_to_RGB(init_F_1[:, :1, :, :])
                    flow_input8 = graytensor_to_RGB(out_F_8[:, :1, :, :])
                    flow0 = model_dic['spynet'](flow_input1, flow_input0)
                    flow8 = model_dic['spynet'](flow_input1, flow_input8)
                    out_F_1 = model_dic['model_flow'](init_F_1, out_F_0, out_F_8, flow0, flow8)
                
                    x_output = out_F_1.squeeze(0).squeeze(0)
                    Prediction_value = x_output.cpu().data.numpy()
                    X_rec = np.clip(Prediction_value[:row, :col], 0, 1)
                    rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
                    rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
                    PSNR_All[j, i] = rec_PSNR
                    SSIM_All[j, i] = rec_SSIM

            result_ave[0, i] = np.mean(PSNR_All[0:args.eva_len, i])
            result_ave[1, i] = np.mean(SSIM_All[0:args.eva_len, i])
            output_data = 'CS ratio is {}, Avg PSNR/SSIM for {} is {:.2f}/{:.4f}' \
                .format(args.sr, cur_video_dir, result_ave[0, i], result_ave[1, i])
            print(output_data)

    result_ave[0, len(video_dir)] = np.mean(result_ave[0, 0:len(video_dir)])
    result_ave[1, len(video_dir)] = np.mean(result_ave[1, 0:len(video_dir)])
    output_data = 'CS ratio is {}, Avg PSNR/SSIM for REDS is {:.2f}/{:.4f}' \
                .format(args.sr, result_ave[0, len(video_dir)], result_ave[1, len(video_dir)])
    print(output_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr', type=float, default=0.070000, help='set sensing rate')
    parser.add_argument('--gop_size', default=8, type=int, help='gop size (default: 8)')
    parser.add_argument('--layer_num', type=int, default=10, help='phase number of the Net')
    parser.add_argument('--eva_len', type=int, default=96, help='evaluation_len')
    main()
