import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNN, self).__init__()
        self.n = 8
        kernel_size = 5
        padding = int((kernel_size - 1) // 2)
        Image_size = (448, 224, 3)
        self.fc1_size = (Image_size[0] // 16) * (Image_size[1] // 16)
        self.conv1 = nn.Conv2d(in_channels=Image_size[2], out_channels=self.n, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 * self.n, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=kernel_size,
                               padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=kernel_size,
                               padding=padding)
        self.fc1 = nn.Linear(self.fc1_size * (8 * self.n), 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):
        """
          prerequisites:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width

          return output, pytorch tensor
          output.Shape == (N,2):
            N := batch size
            2 := same/different pair
        """

        out = self.conv1(inp)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = out.reshape(-1, self.fc1_size * (8 * self.n))
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=1)

        return out


class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()
        self.n = 4
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        Image_size = (224, 224, 6)
        self.fc1_size = (Image_size[0] // 16) * (Image_size[1] // 16)
        self.conv1 = nn.Conv2d(in_channels=Image_size[2], out_channels=self.n, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 * self.n, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=kernel_size,
                               padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=kernel_size,
                               padding=padding)
        self.fc1 = nn.Linear(self.fc1_size * (8 * self.n), 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):
        """
          prerequisites:
          parameter inp: the input image, pytorch tensor
          inp.Shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width

          return output, pytorch tensor
          output.Shape == (N,2):
            N := batch size
            2 := same/different pair
        """
        inp = torch.cat((inp[:, :, :224, :], inp[:, :, 224:, :]), 1)

        out = self.conv1(inp)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = nn.functional.max_pool2d(out, kernel_size=2)

        out = out.reshape(-1, self.fc1_size * (8 * self.n))
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=1)
        return out
