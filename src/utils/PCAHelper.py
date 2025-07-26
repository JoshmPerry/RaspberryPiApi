import numpy as np 
import math
import scipy.io as scio
import os


def load_data(dataloc):
    abs_path = os.path.abspath(dataloc)
    data = scio.loadmat(abs_path)
    return data['A'], [x[0] for x in data['L']]

