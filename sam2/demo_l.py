import cv2
from pathlib import Path
import os
import torch
import sys
sys.path.append("/media/dell/data/zhangyc/regional_registration")# need change

from gluefactory.utils.image import ImagePreprocessor
from gluefactory.utils.tensor import batch_to_device
from gluefactory.eval.io import load_model
from omegaconf import OmegaConf
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from gluefactory.visualization.viz2d import plot_images, plot_keypoints, plot_matches
from copy import deepcopy

from scipy.ndimage import distance_transform_edt,zoom
from scipy import stats

from kmeans_pytorch import kmeans
import kornia
import kornia.utils as KU
import torch.nn.functional as F
from demo_sam1 import SAM2,build_sam
from collections import defaultdict
from scipy.ndimage import label as scipy_label
from skimage import measure

'''
Part of the project code is based on LightGlue and SAM2, and we are grateful for their team's outstanding contributions. Thanks to Tangfei Liao for helping with the LightGlue output debugging. This code can be found in github.  
'''

    
    
