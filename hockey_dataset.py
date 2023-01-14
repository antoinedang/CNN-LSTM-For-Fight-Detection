import json
import torch
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
import torchvision
import cv2

class HockeyDataset(Dataset):
    def __init__(self, device, max_frames, image_size, start_idx=0, end_idx=500):
        super(HockeyDataset, self).__init__()
        self.image_size = image_size
        self.max_frames = max_frames
        self.device = device
        fight_filenames = [f for f in listdir('dataset/fight/') if isfile(join('dataset/fight', f))]
        no_fight_filenames = [f for f in listdir('dataset/no_fight/') if isfile(join('dataset/no_fight', f))]
        self.filenames = fight_filenames[start_idx:end_idx]
        self.filenames.extend(no_fight_filenames[start_idx:end_idx])
        self.num_classes = 2
        self.end_idx = end_idx
        self.start_idx = start_idx

    def preprocessSample(self, video):
        augmentations = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
            ])
        return augmentations(video)
    def video_to_frames(self, video_path, size, max_frames):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            video.release()
            print('Could not open video. Check given path: ' + str(video_path))
            return None
        frames = []
        frameCount = 1 #not 0 since frameStart begins at 1 as well
        while True:
            framesLeft, frame = video.read()
            if framesLeft:
                if max_frames < frameCount: break #do not store frames after our end point
                toTensor = torchvision.transforms.ToTensor()
                frame = toTensor(frame)
                resize = torchvision.transforms.Resize(size)
                frame = resize(frame)
                frames.append(frame)
            else:
                break
        video.release()
        tensor_frames = torch.empty((len(frames), 3, size[0], size[1]))
        for i in range(len(tensor_frames)):
            tensor_frames[i] = frames[i]
        return tensor_frames
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if idx < self.end_idx-self.start_idx:
            filename = 'dataset/fight/' + filename
            label = [1, 0]
        else:
            filename = 'dataset/no_fight/' + filename
            label = [0, 1]
        sample = self.video_to_frames(filename, self.image_size, self.max_frames)
        sample = self.preprocessSample(sample)
        return sample.to(self.device), label