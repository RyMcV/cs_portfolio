# By: Ryan McVicker
# Date started: 06/02/18
# Date completed: xx/xx/xx
# Project: AI Programming with Python nanodegree

## IMPORT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
from collections import OrderedDict
import argparse
import json
import numpy as np
from PIL import Image

## def MAIN
def main():
    # Parse Arguments
    args = get_args()
    
    # Set Variables
    filepath = args.input
    checkpoint = args.checkpoint
    topk = args.top_k
    gpu = args.gpu
    
    # Load category names to variable cat_to_name
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load Checkpoint
    model = load_checkpoint(checkpoint)

    # GPU or CPU
    if gpu and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # Set the model in inference mode with model.eval()
    model.eval()
    
    # Calculate probabilities for Top 5 classes
    prob, idx = predict(filepath, model)
    
    print(prob)
    print(idx)

## def GET_ARGS
def get_args():
    parser = argparse.ArgumentParser(description='parse user input')
    parser.add_argument('input', type=str, help='image file to be classified')
    parser.add_argument('checkpoint', type=str, help='filepath of the checkpoint to be loaded')
    parser.add_argument('--top_k', type=int, help='choose Top K most likely classes, between 1 and 10, as an integer, default is 3', default='3', choices=range(1, 10))
    parser.add_argument('--category_names', type=str, help='JSON filename for mapping categories to real names, default is cat_to_name.json', default='cat_to_name.json')
    parser.add_argument('--gpu', type=bool, help='choose to use GPU, between True or False, default is True', default='True')

## def LOAD_CHECKPOINT
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint('model')
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint('optimizer') 
    model.epochs = checkpoint('epochs')
    model.class_to_idx = checkpoint('class_to_idx')

    return model

## def PREDICT
def predict(filepath, model):
    image_pred = process_image(filepath)
    image_pred = Variable(image_pred.unsqueeze_(0))
    image_pred = image_pred.float()    
    image_pred = image_pred.cuda()
    
    output = model.forward(image_pred)
    ps = torch.exp(output)
    prob, idx = ps.topk(topk)
    
    prob = prob.cpu().numpy()[0]
    idx = idx.cpu().numpy()[0]

    return prob, idx

## def PROCESS_IMAGE
def process_image(filepath):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns an Numpy array
        
    # Open image
    img = Image.open(filepath)
    
    # Scale image
    ratio = img.size[1] / img.size[0]
    if ratio > 1:
        new_x = 256
        new_y = int(ratio * new_x)
        img = img.resize((new_x, new_y))
    else:
        new_y = 256
        new_x = int(new_y / ratio)
        img = img.resize((new_x, new_y))
    
    # Crop image
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    cropped = img.crop(
        (
        half_the_width - 112,
        half_the_height - 112,
        half_the_width + 112,
        half_the_height + 112
        )
                      )

    np_image = np.array(cropped)
    
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255 - np.array(mean)) / np.array(std)
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return torch.from_numpy(np_image).float()    

if __name__ == '__main__':
    main()