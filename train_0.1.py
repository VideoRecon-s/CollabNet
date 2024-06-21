import argparse
import os
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import *
from data_loader import *
from Model.SPYNet import SPyNet
from Model.OCTUF import OCT
from Model.CollabNet import FuNet
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_key = OCT(args.layer_num, args.key_sr).to(device)
    model_nonkey = OCT(args.layer_num, args.nonkey_sr).to(device)
    model_flow = FuNet().to(device)
    spynet = SPyNet().to(device)
    model_dic = {'model_key': model_key, 'model_nonkey': model_nonkey, 'spynet': spynet, 'model_flow': model_flow}

    all_parameters = []
    for i in model_dic:
        all_parameters += list(model_dic[i].parameters())

    scaler = GradScaler()
    optimizer = optim.AdamW(all_parameters, lr=args.lr)
    train_set = REDS_train(gop_size=args.gop_size, image_size=args.image_size,load_filename='REDS.npy')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=False)
    criterion = loss_fn

    spynet.load_state_dict(torch.load('./CheckPoint/spynet.pth'), strict=False)
    pth1_dir = "./CheckPoint/OCT_050.pth"
    checkpoint1 = torch.load(pth1_dir)
    model_key.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint1['net'].items()})
    pth2_dir = "./CheckPoint/OCT_010.pth"
    checkpoint2 = torch.load(pth2_dir)
    model_nonkey.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint2['net'].items()})

    model_dir = "./CheckPoint/ratio_{}_{}".format(args.key_sr, args.nonkey_sr)
    if not os.path.exists(model_dir):os.mkdir(model_dir)

    if args.start_epoch > 0:
        checkpoint = torch.load("{}/params_{}.pth".format(model_dir, args.start_epoch), map_location=device)
        model_flow.load_state_dict(checkpoint['net_flow'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = args.start_epoch + 1

    print('Start of training')
  
    for epoch in range(start_epoch, args.epochs + 1):

        for i in model_dic:
            if i == 'model_flow':
                model_dic[i].train()
            else:
                model_dic[i].eval()
        sum_loss = 0

        for i, inputs in enumerate(train_loader):
            F_0 = inputs[:, 0, :, :, :].to(device)
            F_1 = inputs[:, 1, :, :, :].to(device)
            F_8 = inputs[:, 2, :, :, :].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                out_F_0 = model_dic['model_key'](F_0)
                out_F_8 = model_dic['model_key'](F_8)
                init_F_1 = model_dic['model_nonkey'](F_1)
                flow_input0 = graytensor_to_RGB(out_F_0[:, :1, :, :])
                flow_input1 = graytensor_to_RGB(init_F_1[:, :1, :, :])
                flow_input8 = graytensor_to_RGB(out_F_8[:, :1, :, :])
                flow0 = model_dic['spynet'](flow_input1, flow_input0)
                flow8 = model_dic['spynet'](flow_input1, flow_input8)
                
            out_F_1 = model_dic['model_flow'](init_F_1, out_F_0, out_F_8, flow0, flow8)
            
            recnLoss_F1 = criterion(out_F_1, F_1)
            loss = recnLoss_F1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sum_loss += loss.item()

        print_data = "[{}/{}] Loss: {}".format(epoch, args.epochs, sum_loss)
        print(print_data)

        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'net_flow': model_flow.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, "{}/params_{}.pth".format(model_dir, epoch))

    print('End of training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key_sr', type=float, default=0.500000, help='set sensing rate')
    parser.add_argument('--nonkey_sr', type=float, default=0.10000, help='set sensing rate')
    parser.add_argument('--start_epoch', default=0, type=int, help='epoch number of start training')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size') 
    parser.add_argument('--image_size', type=int, default=160, help='image size for train')
    parser.add_argument('--gop_size', default=8, type=int, help='gop size (default: 8)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--layer_num', type=int, default=10, help='phase number of the Net')
    main()
