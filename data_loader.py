import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop
from PIL import Image


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class REDS_train(Dataset):
    def __init__(self, gop_size=8, image_size=96, load_filename='REDS.npy'):

        self.load_filename = load_filename
        self.fnames = np.load(self.load_filename)
        self.fnames = self.fnames.tolist()
        self.gop_size = gop_size
        self.image_size = image_size

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        video_path = self.fnames[idx]
        video_x = self.get_single_video_x(video_path)
        return video_x

    def get_single_image(self, image_path):
        image = load_img(image_path)
        input_compose = Compose([CenterCrop(160),ToTensor()])
        image = input_compose(image)
        return image

    def construct_sample(self, file_dir, image_id):
        image_index_i = random.randint((image_id + 1), (image_id + self.gop_size - 1))
        s_list = ["{:0>8}".format(image_id), "{:0>8}".format(image_index_i), "{:0>8}".format(image_id + self.gop_size)]
        for i, s in enumerate(s_list):
            image_name = s + '.png'
            image_path = os.path.join(file_dir, image_name)
            single_image = self.get_single_image(image_path)
            if i == 0:
                train_x = single_image
            else:
                train_x = torch.cat((train_x, single_image), dim=0)
        train_x = self.augment(train_x)
        return train_x

    def get_single_video_x(self, file_dir):
        a_file = os.listdir(file_dir)
        frame_count = len(a_file)
        image_start = random.randint(0, frame_count - self.gop_size - 1)
        train_x = self.construct_sample(file_dir, image_start)

        return train_x
    
    def augment(self, train_x, hflip=True, vflip=True, rot=True):

        _, h, w = train_x.shape
        th = tw = self.image_size

        j = random.randint(0, w - tw)
        i = random.randint(0, h - th)
        train_x = train_x[:, i:i + th, j:j + tw]

        hflip = hflip and random.random() < 0.5
        vflip = vflip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            train_x = train_x.flip([-1])
        if vflip:
            train_x = train_x.flip([-2])
        if rot90:
            train_x = train_x.permute(0, 2, 1)

        train_x = train_x.unsqueeze(1)
        return train_x