import torch
from torch import nn
import torch.nn.functional as F

class Classifier_FCN(nn.Module):

    def __init__(self, input_shape, nb_classes, filter_count):
        super(Classifier_FCN, self).__init__()
        self.nb_classes = nb_classes
        self.filter_count = filter_count

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.filter_count, kernel_size=8, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(self.filter_count,)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count*2, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(self.filter_count*2)


        self.conv3 = nn.Conv1d(in_channels=self.filter_count*2, out_channels=self.filter_count, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(self.filter_count)

        # self.relu = nn.ReLU()

        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.filter_count, self.nb_classes)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avgpool1(x)
        x = self.flatten(x)
        # x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x