import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    def __init__(self, config):
        super(BaseLoss, self).__init__()
        """ 
        config is a dictionary containing all the parameters relevant to training
        """
        self.config = config

    def forward(self, y_hat, y):
        raise NotImplementedError
