import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
 
# a special module that converts [batch, channel, w, h] to [batch, units]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
 
# assuming input shape [batch, 3, 64, 64]
cnn = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=2048, kernel_size=(3,3)),
    nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(3,3)),
    nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3,3)),
    nn.ReLU(),
    nn.MaxPool2d((6,6)),
    nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(20,20)),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(20,20)),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(20,20)),
    nn.Softmax(),
    Flatten(),
    nn.Linear(64, 256),
    nn.Softmax(),
    nn.Linear(256, 10),
    nn.Sigmoid(),
    nn.Dropout(0.5)  
)
