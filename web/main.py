from __future__ import print_function, division
from flask import Flask, render_template, request
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import Optional
from fastapi import FastAPI
from datetime import datetime
import requests
import torch
import base64
import cv2
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


app = Flask(__name__)

def ToTensor(image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image1 = image.transpose((2, 0, 1))
        return (torch.FloatTensor(image1).unsqueeze(0))

class yoyoBSModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.norm5 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 3)
        self.norm6 = nn.BatchNorm2d(32)
        self.fc6 = nn.Linear(3872, 2)
        self.flatten = nn.Flatten()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.ReLU(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.norm4(x)
        x = self.ReLU(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.norm6(x)
        x = self.ReLU(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc6(x)
        return x

@app.route('/', methods=['GET'])
# @app.get("/") if using Pycharm in old-version
def root():
    return render_template('webcam2.html')

@app.route('/', methods=['POST'])
# @app.post("/") if using Pycharm in old-version
def root_post():
    json.loads(request.data)
    data = json.loads(request.data)
    base64.b64decode(data['image'])
    decode = base64.b64decode(data['image'])
    nparr = np.fromstring(decode, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img.shape)
    imageTensor = ToTensor(img)

    PATH = './yoyomodel.pt'
    model = torch.load(PATH, map_location=torch.device('cpu')) # using CPU
    model.eval()
    with torch.no_grad():   # Do not upgrade grad
        outputs = model(imageTensor)
        outputs = torch.softmax(outputs,dim=1)
        print(outputs)
        decision = outputs[0][0].numpy().item(0)  # prob. of small-size frames
        print(decision)
        if decision<0.3 : # Prob. of small-size frames is low, return big-size
            data = 1
        elif decision<0.7:  # Prob. of small-size frames is medium, return small-size & big-size
            data = 2
        else:            # Prob. of small-size frames is high, return small-size
            data = 0
        print(data)
        return str(data)

if __name__ =='__main__':
    app.run('127.0.0.1', 8010, debug=True)