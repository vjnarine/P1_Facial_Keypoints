## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        
        self.linear1 = nn.Linear(in_features=36864, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x= F.max_pool2d(F.relu(self.conv1(x)),2)
        x= F.max_pool2d(F.relu(self.conv2(x)),2)
        x= F.max_pool2d(F.relu(self.conv3(x)),2)
       

        x = x.view(-1, 36864)
        x= F.relu(self.linear1(x))
        x = F.dropout(x, p=0.3)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=0.2)
        x = self.linear3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
