import sys
import scipy.io
import h5py
import numpy as np
import os, sys, time, subprocess, h5py, argparse, logging, pickle
import numpy as np
import pandas as pd
from os.path import join as oj
from copy import deepcopy
from scipy.ndimage import imread
import skimage
from PIL import Image
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path                   
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import data.data as data
import re
import seaborn as sns
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from tqdm import tqdm
import scipy.optimize as opt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

def plot_all(ims, save_dir, normalize=True):
    plt.figure(figsize=(12, 12), dpi=300, facecolor='white')
    R, C = 6, 6
    ims = np.array([ims[i] for i in range(len(ims))])
    vmin, vmax = np.min(ims), np.max(ims)
    for i in range(len(ims)):
        plt.subplot(R, C, i+1)
        im = ims[i]
        if normalize:
            plt.imshow(im, vmin=vmin, vmax=vmax) #, cmap='gray_r')
        else:
            plt.imshow(im) #, cmap='gray_r')
        plt.axis('off')
    plt.subplots_adjust(hspace=0, wspace=0)
    if normalize:
        plt.savefig(oj(save_dir, 'summ_normalized.png'), bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(oj(save_dir, 'summ_unnormalized.png'), bbox_inches="tight", pad_inches=0)
    plt.show()

    
def plot_images(ax, xData, yData, ims, width_physical=None, labs=None, normalize=True):
    if labs is None:
        labs = ["" for _ in range(len(ims))]
    if normalize:
        vmin, vmax = np.min(ims), np.max(ims)
        ims_plot = (ims - vmin) / (vmax - vmin) # normalize
        ims_plot[:, 0, 0] = 1 # keep topleft pixel 1 to enforce normalization

    for idx, (x, y) in enumerate(zip(xData, yData)):
        if width_physical is None:
            small_im_dim = 0.5
        else:
            small_im_dim = width_physical[idx] / 8
        x -= small_im_dim / 2
        y -= small_im_dim / 2
        bb = Bbox.from_bounds(x, y, small_im_dim, small_im_dim)
        bb2 = TransformedBbox(bb, ax.transData)
        bbox_image = BboxImage(bb2, norm = None,
                                  origin=None, clip_on=False) #, cmap='gray_r')
        plt.title(labs[idx])
        bbox_image.set_data(ims_plot[idx])
        bbox_image.set_alpha(1.0)
        ax.add_artist(bbox_image)
        
        
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim(-0.25, max(xData) + 0.5 / 2)
    plt.ylim(-0.25, max(yData) + 0.5 / 2)    