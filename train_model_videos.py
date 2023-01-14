from torchvision import models
from torch import nn
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from ignite.metrics import Accuracy, Recall, Precision
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.engine.events import Events
from os import listdir
from os.path import isfile, join
import re
from hockey_dataset import HockeyDataset
from CNN2RNN import RecursiveCNN

def start_training(train_loader, test_loader, epochs, model, optimizer, loss, metrics, gradient_accumulation_steps): 
        trainer = create_supervised_trainer(model, optimizer, loss, gradient_accumulation_steps=gradient_accumulation_steps)
        evaluator = create_supervised_evaluator(model, metrics=metrics)
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_batch_complete():
            print(f"epoch {trainer.state.epoch} iteration {trainer.state.iteration} completed.")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch_time():
            print(f"epoch {trainer.state.epoch} completed, time elapsed : {trainer.state.times['EPOCH_COMPLETED']}")
            evaluate()

        def evaluate():
            print("evaluating model")
            evaluator.run(test_loader)
            metrics = dict(evaluator.state.metrics)
            print(str(metrics))
            torch.save(model.state_dict(), 'models/model_e' + str(trainer.state.epoch) + "_accuracy_" + str(metrics['accuracy']) + ".pt")

        trainer.run(train_loader, max_epochs=epochs)
        evaluator.run(test_loader)
        return 
        
def custom_collate_fn(data): #data is a list of [batch_size] tuples of the form (video, label), we need to return the padded videos and the corresponding labels as tensors
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    videos = []
    labels = torch.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        sample = data[i]
        videos.append(sample[0])
        labels[i] = torch.tensor(sample[1])
    videos = rnn_utils.pad_sequence(videos, batch_first=True, padding_value=0.0)
    return videos.to(device), labels.to(device)

def getMetricInFilename(f):
    metric_in_filename = re.findall('\d*\.?\d+',f)[1]
    return metric_in_filename

def roundToLabel(output):
    y_pred, y = output
    return y_pred, y

def launch(data_splits, batch_sizes, max_frames, device, profile, epochs, image_size, gradient_accumulation_steps, force_generate_dataset, continuous_training):
    print("starting on " + str(device))
    #load dataset from file system or from scratch
    try:
        if force_generate_dataset: raise Exception
        train_ds, test_ds, val_ds = torch.load("data/datasets.sav")
        print("loaded datasets from file system")
    except Exception as e:
        print("generating dataset from scratch")
        #there are 1000 videos (500 per label) so stratified split is easy
        split_per_label = [500*x for x in data_splits]
        train_ds = HockeyDataset(device, max_frames, image_size, 0, int(split_per_label[0]))
        test_ds = HockeyDataset(device, max_frames, image_size, int(split_per_label[0]), int(split_per_label[0]+split_per_label[1]))
        val_ds = HockeyDataset(device, max_frames, image_size, int(split_per_label[0]+split_per_label[1]), int(split_per_label[0]+split_per_label[1]+split_per_label[2]))
        torch.save((train_ds, test_ds, val_ds), "data/datasets.sav")

    print("setting up dataloaders, metrics and model")
    # split into 3 dataloaders (for train, test, and valid)
    train_dl = DataLoader(train_ds, batch_size = batch_sizes[0], shuffle=True, collate_fn= custom_collate_fn, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size = batch_sizes[1], shuffle=True, collate_fn= custom_collate_fn, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size = batch_sizes[2], shuffle=True, collate_fn= custom_collate_fn, pin_memory=True)
    #load model
    model = RecursiveCNN(train_ds.num_classes)
    if continuous_training:
        files = [f for f in listdir('models/') if isfile(join('models/', f))]
        noModels = False
        if len(files) == 0:
            print("No model file to continue training from.")
            no_models = True
        if not no_models:
            best_model = max([getMetricInFilename(f) for f in files])
            for f in files:
                if best_model not in f: continue
                model.load_state_dict(torch.load(f))    
                print("continuing from: " + f)
    model.to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    metrics = {'accuracy': Accuracy(device=device, output_transform=roundToLabel), 'recall': Recall(device=device), 'precision':Precision(device=device)}

    print("Starting training with " + str(len(train_ds)) + " out of 1000 samples.")
    if profile:
        with torch.autograd.profiler.profile() as prof:
            try:
                measured_metrics = start_training(train_dl, test_dl, epochs, model, optimizer, loss, metrics, gradient_accumulation_steps)
            except Exception as e:
                print("ended with error " + str(e))
            print(prof.key_averages())
    else:
        measured_metrics = start_training(train_dl, test_dl, epochs, model, optimizer, loss, metrics, gradient_accumulation_steps)
    
    print("done!")

if __name__ == '__main__':
    data_splits = [0.6, 0.2, 0.2]
    batch_sizes = [16, 16, 8]
    gradient_accumulation_steps = 1
    image_size = (224, 224)
    max_frames = 20 #no more than 20 frames
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    epochs = 50
    profile = False
    force_generate_dataset = False #when True re-generates the dataset object instead of loading from file system
    continuous_training = False
    launch(data_splits, batch_sizes, max_frames, device, profile, epochs, image_size, gradient_accumulation_steps, force_generate_dataset, continuous_training)
