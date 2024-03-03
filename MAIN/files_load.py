import numpy as np
#import random
#import math
#import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook
#import copy
#from model_control import *
#import pickle
#from ANN import *

'''
with open("./train-images.idx3-ubyte","rb") as f :
  f.read(16)
  train_img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)
with open("./train-labels.idx1-ubyte","rb") as f :
  f.read(8)
  train_lab = np.fromfile(f,dtype=np.uint8)

'''
with open("./t10k-images.idx3-ubyte","rb") as f :
  f.read(16)
  test_img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)
with open("./t10k-labels.idx1-ubyte","rb") as f :
  f.read(8)
  test_lab = np.fromfile(f,dtype=np.uint8)


