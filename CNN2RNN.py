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
        
        self.resnet_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        #we get rid of the fully connected layer at the end of the architecture so we can put our decoder and LSTM instead
        self.rnn = nn.LSTM(input_size=self.resnet_cnn.fc.out_features, hidden_size=512, num_layers=20)
        #apply dropout to decrease the chance of overfitting
        self.final_fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_classes))
    def forward(self, x): #input should be a batch of videos of below shape
        b_z, length, colors, height, width = x.shape
        hidden_state = None
        for frame in range(length):#for every image in the video (all samples in the batch at once)
            resnet_out = self.resnet_cnn((x[:,frame]))
            lstm_out, hidden_state = self.rnn(resnet_out.unsqueeze(0), hidden_state) #rnn takes input of shape (sequence_length, batch_size, input_size)
        out = self.final_fc(lstm_out.squeeze(0))
        return torch.round(out)