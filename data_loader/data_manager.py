

import os
import torch as th
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from utils import utils

from torch.utils import data
import numpy as np
import random

from skimage import io, transform

#####################################################################################################
#####################################################################################################

class DataManager(data.Dataset):
    def __init__(self, args, params, data):

        self._classes = args.classes
        self._data = data.data
        self._label = data.label

        #print(self._data.shape[0])
        print("Number of spectra in the data set", self._label.shape[0])

    def sequence_length(self):
        return self._label.shape[0]

    def __len__(self):
        return self._label.shape[0]

    def __getitem__(self, idx):

        ## load data ##
        # self._labels_frame.iloc[idx, j], row  idx, column j of csv file
        # make sure you kow what written in the csv, and create input and label accordingly

        # Here,is Output:
        #                 - Image
        #                 - Label



        ###################################################################################
        ###################################################################################
        ## Image ##

        image  = self._data[idx,:,:]
        label = self._label[idx,:]

        image = np.transpose(image,(1,0))


        return image, label




## Preaload all the data
class LoadData:
    def __init__(self, args,  csvpath):



        str_data  = os.path.join(args.path,'x'+csvpath)
        str_label = os.path.join(args.path,'y'+csvpath)

        data  = np.loadtxt(str_data) #pd.read_csv(str_data)
        self.label = th.tensor(np.loadtxt(str_label, delimiter=";"),dtype=th.float32) #pd.read_csv(str_label)

        self.data = th.tensor(np.reshape(data,(-1,190,3)),dtype=th.float32)
