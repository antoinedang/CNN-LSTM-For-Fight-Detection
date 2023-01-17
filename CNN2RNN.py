from torchvision import models
from torch import nn
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import ignite
from ignite.metrics import Accuracy, Recall, Precision
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.engine.events import Events
from os import listdir
from os.path import isfile, join
import re
import cv2

class RecursiveCNN(nn.Module):
    def __init__(self, num_classes):
        super(RecursiveCNN, self).__init__()
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.feature_extractor = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

        self.rnn = nn.LSTM(input_size=1000, hidden_size=512, num_layers=20)
        
        self.final_fc = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_classes),
            nn.Softmax(dim=1))
    def forward(self, x): #input should be a batch of videos of below shape
        b_z, length, colors, height, width = x.shape
        hidden_state = None
        for frame in range(length):#for every image in the video (all samples in the batch at once)
            cnn_out = self.feature_extractor((x[:,frame]))
            lstm_out, hidden_state = self.rnn(cnn_out.unsqueeze(0), hidden_state) #rnn takes input of shape (sequence_length, batch_size, input_size)
        out = self.final_fc(lstm_out.squeeze(0))
        return out