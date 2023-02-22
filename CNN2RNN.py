from torchvision import models
from torch import nn
import torch

class RecursiveCNN(nn.Module):
    def __init__(self, num_classes):
        super(RecursiveCNN, self).__init__()
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        self.feature_extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.rnn = nn.LSTM(input_size=1000, hidden_size=512, num_layers=3)
        
        self.final_fc = nn.Sequential(
            nn.Linear(512, 50),
            nn.Sigmoid(),
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