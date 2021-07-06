import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import re
import yaml

loader = yaml.SafeLoader
# redefine regex for yaml loader so scientific-notation is read as floating-point
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def load_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=loader)
    return data

def sec_to_dhm_str(num_seconds):
    """Converts number of seconds to number of days, hrs, minutes, and seconds"""
    num_seconds = int(num_seconds)

    num_days = num_seconds // 86400
    num_seconds = num_seconds % 86400

    num_hrs = num_seconds // 3600
    num_seconds = num_seconds % 3600

    num_minutes = num_seconds // 60
    num_seconds = num_seconds % 60

    result = "{:02d}d{:02d}hr{:02d}m{:02d}s".format(num_days, num_hrs,
    num_minutes, num_seconds)

    return result