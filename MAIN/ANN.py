import numpy as np
#import random
#import math
#import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook
#import pickle
#import copy
import torch
from torch import optim
from torch import nn
#from torchvision import datasets
#from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
#from get_hand_write import *

#onehot = np.identity(10)

"""
#激活函數
def bypass(x) :
  return x
def tanh(x) :
  return np.tanh(x)
def sigmoid(x) :
  return 1/(1+np.exp(x))
def softmax(x) :
  exp = np.exp(x - x.max())
  return exp/exp.sum()
def leaky_relu(x) :
   alpha = 0.1
   return np.maximum(x,x * alpha)


#損失函數
def loss(y,p) :
  return ((y-p)**2).sum()

#differential
def d_bypass(x) :
  return x
def d_tanh(x) :
  return 1/np.cosh(x)**2
def d_sigmoid(x) :
  return -np.exp(x)/((1+np.exp(x))**2)
def d_softmax(x) :
  sm = softmax(x)
  return np.diag(sm) - np.outer(sm,sm)
def d_loss(y,p) :
  return 2*(p-y)
def d_leaky_relu(x) :
  alpha = 0.1
  return np.where(x>=0,1,alpha)


differential = {bypass:d_bypass,tanh:d_tanh,softmax:d_softmax,loss:d_loss,sigmoid:d_sigmoid,leaky_relu:d_leaky_relu}
out_type = {bypass:"times",tanh:"times",softmax:"dot",sigmoid:"times",leaky_relu:"times"}

class ANN :
  def __init__(self,neuron_input,layer=[]) :
    self.layer = layer
    if len(layer) > 0 :
      self.last_neuron = len(layer[-1][1])
    else :
      self.last_neuron = neuron_input

  def neuron_init_fuc(self,y) :
    x = self.last_neuron
    return (-math.sqrt(6/(x+y)),math.sqrt(6/(x+y)))

  def add_layer(self,neuron,neuron_init,bias_init,activation) :
    front_neuron = self.last_neuron
    self.layer.append( [np.random.random(front_neuron*neuron).reshape(front_neuron,neuron)*(neuron_init[1]-neuron_init[0])+neuron_init[0],
            np.random.random(neuron)*(bias_init[1]-bias_init[0])+bias_init[0],
            activation ])
    self.last_neuron = neuron
  def gradient(self,data,sample,loss_fuc) :
    output_un_act = []
    output_act = []
    gradient_layer = [None]*len(self.layer)
    for i in range(len(self.layer)) :
      output_act.append(data)
      un_act = np.dot(data,self.layer[i][0]) + self.layer[i][1]
      data = self.layer[i][2](un_act)
      output_un_act.append(un_act)


    #Loss求偏导
    d_bias = differential[loss_fuc](sample,data)


    for i in range(len(self.layer)-1,-1,-1) :
      if out_type[self.layer[i][2]] == "times" :
        d_bias = differential[self.layer[i][2]](output_un_act[i]) * d_bias
      elif out_type[self.layer[i][2]] == "dot" :
        d_bias = np.dot(differential[self.layer[i][2]](output_un_act[i]),d_bias)
      gradient_layer[i] = (np.outer(output_act[i],d_bias),d_bias)
      d_bias = np.dot(self.layer[i][0],d_bias)

    return gradient_layer

  def model_loss(self,input,sample,loss_fuc) :
    return loss_fuc(sample,self.run(input))

  def train(self,train_data,train_sample,loss_fuc,learning_rate) :
    n = len(train_data)
    for data_index in range(len(train_data)) :
      data = train_data[data_index]
      sample = train_sample[data_index]
      gradient = self.gradient(data,sample,loss_fuc)
      for i in range(len(self.layer)) :
        self.layer[i][0] = self.layer[i][0] - gradient[i][0]*learning_rate/n
        self.layer[i][1] = self.layer[i][1] - gradient[i][1]*learning_rate/n
  def train_layer(self,train_data,train_sample,loss_fuc,learning_rate,in_layer) :
    n = len(train_data)
    for data_index in range(len(train_data)) :
      data = train_data[data_index]
      sample = train_sample[data_index]
      gradient = self.gradient(data,sample,loss_fuc)
      self.layer[in_layer][0] = self.layer[in_layer][0] - gradient[in_layer][0]*learning_rate/n
      self.layer[in_layer][1] = self.layer[in_layer][1] - gradient[in_layer][1]*learning_rate/n

    return 1
  def run(self,input,show_output=False,img_show=False) :
    for i in range(len(self.layer)) :
      input = self.layer[i][2](np.dot(input,self.layer[i][0]) + self.layer[i][1])
      if(show_output) :
        print(f"{i+1}---->: {input}")
      if(img_show) :
        Fix = plt.figure()
        size = int(math.sqrt(len(input)))
        plt.imshow(input[:size*size].reshape(size,size))

    return input


def save_model(model,name) :
  with open(f"./{name}","wb") as f :
    pickle.dump(model.layer,f)
def load_model(name) :
  f = open(f"./{name}","rb")
  return ANN(0,pickle.load(f))

"""

class LeNet5(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1, 6, 5, padding=2),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),
      nn.Conv2d(6, 16, 5),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),
      nn.Flatten(),
      nn.Linear(16*5*5, 120),
      nn.ReLU(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, 10)
    )

  def forward(self, x):
    output = self.model(x)
    return output

class CNN1(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1, 6, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(6, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )

  def forward(self, x):
    output = self.model(x)
    return output

class CNN2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
    self.relu2 = nn.ReLU()
    self.pool1 = nn.MaxPool2d((2, 2))
    self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
    self.relu3 = nn.ReLU()
    self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
    self.relu4 = nn.ReLU()
    self.pool2 = nn.MaxPool2d((2, 2))
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(64 * 7 * 7, 64 * 7)
    self.relu5 = nn.ReLU()
    self.fc2 = nn.Linear(64 * 7, 120)
    self.relu6 = nn.ReLU()
    self.fc3 = nn.Linear(120, 10)
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool1(x)
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu5(x)
    x = self.fc2(x)
    x = self.relu6(x)
    return self.fc3(x)

class LECNN2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
    self.relu2 = nn.ReLU()
    self.pool1 = nn.MaxPool2d((2, 2))
    self.conv3 = nn.Conv2d(16, 32, 5)
    self.relu3 = nn.ReLU()
    self.pool2 = nn.MaxPool2d((2, 2))
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32 * 5 * 5, 84)
    self.relu4 = nn.ReLU()
    self.fc2 = nn.Linear(84, 16)
    self.relu5 = nn.ReLU()
    self.fc3 = nn.Linear(16, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.pool1(x)
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu4(x)
    x = self.fc2(x)
    x = self.relu5(x)
    x = self.fc3(x)
    return x
