#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ipdb

class net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.conv2d(1, 32, 5, 2, padding=2)
        self.conv2 = nn.conv2d(16, 64, 5, 2, padding=2)
        self.conv3 = nn.conv2d(64, 32, 5, 2, padding=2)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        # initialize the parameters for [self.fc1, self.fc2, self.fc3]
        nn.init.normal_(self.fc1.weight, 0.0, 1 / sqrt(512))
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, 0.0, 1 / sqrt(64))
        nn.init.constant_(self.fc2.bias, 0.0)

        nn.init.normal_(self.fc3.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc3.bias, 0.0)

        nn.init.normal_(self.fc4.weight, 0.0, 1 / sqrt(8))
        nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)

def train_epoch(data_loader, model, criterion, optimizer):

    for batch_idx, (data, target) in enumerate(data_loader):
        pass


    




def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    #image_dim = config('image_dim')
    image_dim = 32
    return imresize(arr=X, size=(image_dim, image_dim), interp='bicubic')
    #resized = np.ndarray(shape=(len(X), image_dim, image_dim), dtype='uint8')
    #for i, image in enumerate(X):
    #    resized[i] = imresize(arr=image, size=(image_dim, image_dim), interp='bicubic')

    #ipdb.set_trace()
    #return resized

def show_np_image(X):
    plt.imshow(X)
    plt.show()

def show_image(img_name):
    plt.imshow(io.imread(os.path.join('data/', img_name)))
    plt.show()

class SpaceDataset(Dataset):
    "Solar wind dataset"

    def __init__(self, transform=None):
        self.landmarks_frame = pd.read_csv('labels.csv')
        self.root_dir = 'data/'
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        #ipdb.set_trace()
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 1])
        image = io.imread(img_name)
        density = self.landmarks_frame.iloc[idx, 2]

        image = resize(image[:,:,0])

        sample = {'image': image, 'density': density}

        return sample


def main():
    d = SpaceDataset()
    
    for i in range(len(d)):
        #ipdb.set_trace()
        sample = d[i]
        show_np_image(sample['image'])
        print(sample)


    ipdb.set_trace();
    print('done')




    #frame = pd.read_csv('labels.csv')
    #ipdb.set_trace()

    #n_train = 5000
    #n_test = 1000

    #img_name = frame.iloc[n_train, 1]
    #value = frame.iloc[n_train, 2]

    #print('done')


if __name__ == '__main__':
    main()
