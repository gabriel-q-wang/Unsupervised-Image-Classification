import numpy as np
import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path)
sys.path.append(os.path.dirname(curr_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_loss import BaseLoss

class SupervisedLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(SupervisedLoss, self).__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_hat, y):
        y_hat = self.softmax(y_hat)
        loss = self.loss_fn(y_hat, y)
        return loss


if __name__ == "__main__":
    loss_fn = SupervisedLoss("fake_config")
    print("done")
