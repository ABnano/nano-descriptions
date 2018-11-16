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

import warnings
warnings.filterwarnings("ignore")
from lib.igg import binarywave

# dataset to access conditions / images
class Star_dset_mid():
    def read_conditions(self, fname):
        c_raw = pd.read_csv(fname, delimiter=',')
        c = pd.DataFrame() #pd.read_csv(oj('data', 'star_polymer_conditions.csv'), delimiter=',')
        c['fnames'] = c_raw.fnames.astype(str)
        c['conc'] = c_raw.conc_solution # mol/L
        c['solvent_glyc'] = [int(x[:2]) for x in c_raw.solvent_glyc]# percent glycerol, 5% ethanol, 85% water
        c['pressure'] = c_raw.pressure # mbar
        c['time'] = c_raw.time # time delivered for (s)
        c['zmin'] = zz = [float(re.sub("[^0-9*.-]", "", zz).split('*')[0]) for zz in c_raw.zz] # physical width = physical height# max zscale
        c['zmax'] = zz = [float(re.sub("[^0-9*.-]", "", zz).split('*')[1]) for zz in c_raw.zz] # physical width = physical height# max zscale
        c['width_physical'] = [float(re.sub("[^0-9*.]", "", xy).split('*')[0]) for xy in c_raw.xy] # physical width = physical height
        c['R'] = c_raw.R # based on scan lines
        c['C'] = c_raw.C # based on scan lines
        return c


    def read_ims(self, data_raw_dir, conditions):
        ims = []
        for fname in conditions['fnames']:
            try:
                data = binarywave.load(oj(data_raw_dir, fname + '.ibw'))['wave']
                ims.append(data['wData'][..., -1])
            except:
                print(fname)
        conditions['ims'] = ims
        return conditions

    def preprocess_ims(self, dset):
        # images 3 and 33 have strange artifacts which require removal
        dset = dset.drop([3, 33], axis=0).reset_index()

        # preprocess images
        ims = deepcopy(dset.ims)
        for i in range(len(ims)):
            im = skimage.transform.resize(ims[i], (512, 512))
            im -= np.min(im)
            ims[i] = im

        # outliers (indexes after dropping)
        ims[21][ims[22] < 1e-8] = 1.17925154e-08 # image 21 has a couple very low points that ruin its contrast

        dset.ims = ims
        return dset
    

# dataset to access conditions / images
class Star_dset_orig(Dataset):
    def __init__(self, star_dir='star_polymer'):
        self.conditions = pd.read_csv(oj(star_dir, 'star_polymer_conditions.csv'), delimiter=',')
        self.ims = [self.read_and_crop_tif(oj(star_dir, fname)) for fname in self.conditions['im_fname']]
        self.star_dir = star_dir

    def read_and_crop_tif(self, fname):
        im = Image.open(oj(fname + '.TIF'))
        imarray = np.array(im)[:, :, 0] # convert to grayscale
        # im_downsample = imresize(imarray, size=(imarray.shape[0]//8, imarray.shape[1]//8)) # downsample by 8
        im_downsample = imresize(imarray, size=(764, 915)) # downsample by 8
        im_cropped = im_downsample[20: 695, 102: 777]
        return im_cropped
    
    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        conditions_dict = self.conditions.loc[0].to_dict()
        im_dict = {'im': self.ims[idx]}
        return {**conditions_dict, **im_dict}

    
# dataset to access just images    
class Mixed_sam_dset_orig(Dataset):
    def __init__(self, mixed_sam_dir='mixed_sam'):
        fnames = os.listdir(mixed_sam_dir)
        self.ims = [self.read_and_crop_tif(oj(mixed_sam_dir, fname)) for fname in fnames if '.TIF' in fname]
        self.mixed_sam_dir = mixed_sam_dir

    def read_and_crop_tif(self, fname):
        im = Image.open(oj(fname))
        imarray = np.array(im)[:, :, 0] # convert to grayscale
        im_downsample = imresize(imarray, size=(imarray.shape[0]//8, imarray.shape[1]//8)) # downsample by 8
        im_cropped = im_downsample[15: 338, 73: 396]
        return im_cropped
    
    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        return {'im': self.ims[idx]}