import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
from model import Model
from custom_dataset import *

if __name__ == "__main__":
    yaml_path = "configs/infer_config.yaml"
    model = Model(yaml_path)
    model.inference()
