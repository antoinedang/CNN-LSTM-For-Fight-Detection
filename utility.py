import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from wlasl_dataset import WLASLDataset
from random import randrange

encodingVector = []

def export_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frame_array:
        out.write(frame)
    out.release()

def visualizeVideo(video):
    deNormalize = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1.0/0.22803, 1.0/0.22145, 1.0/0.216989 ]),
        torchvision.transforms.Normalize(mean = [ -0.43216, -0.394666, -0.37645 ], std = [ 1., 1., 1. ]) ])
    video = deNormalize(video)
    img = torchvision.utils.make_grid(video)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

if __name__ == '__main__':  
    ds = torch.load("data/full-dataset.sav")
    vid = ds[randrange(len(ds))][0]
    print(vid[0][0][0][0])
    print(vid.shape)
    visualizeVideo(vid)


# -m torch.utils.bottleneck