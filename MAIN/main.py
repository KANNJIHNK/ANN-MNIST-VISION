import time

import numpy as np
import random
from files_load import *
from ANN import *
import torch
#from torch import optim
#from torch import nn
#from torchvision import datasets
#from torch.utils.data import DataLoader, TensorDataset
#import torch.nn.functional as F
from get_hand_write import *


# # Load model
#
# an = load_model("784_leaky_relu_tanh_1")
# #784_leaky_relu_tanh_1
# #lr_tanh_1
#
#
#

# train = datasets.MNIST('data/', download=True, train=True)
# test = datasets.MNIST('data/', download=True, train=False)
#
# X_train = train.data.unsqueeze(1)/255.0
# y_train = train.targets
# trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
#
# X_test = test.data.unsqueeze(1)/255.0
# y_test = test.targets

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(device)

# an = LeNet5()
an = LECNN2()
an.load_state_dict(torch.load("lecnn-4-line-3-1.pt",map_location=device))
an.train(False)

#钩子
def hook(module,fin,fout) :
    output_list.append(fout.data.detach().numpy())
    return None

children = an.children()
output_list = []
for child in children :
    child.register_forward_hook(hook)


def show_output() :
    fuc_list = list(an.children())
    plt.figure(figsize=(5, 5))
    for i in range(10):
        print(f"in:{i},fuc:{fuc_list[i]}")
        show_img = output_list[i][0][1].detach().numpy()
        print(show_img.shape)
        plt.subplot(2, 5, i+1)
        plt.imshow(show_img, cmap="gray")
    plt.show()
        #time.sleep(5)


screen,img = create_board()
is_change = True
auto_show = True
gochange = False

wait_time = 3
last_time = 0

pred = an(torch.Tensor(img.reshape(1,1,28,28),device=device)/255.0).argmax()
img, is_change,auto_show = update_screen(screen, img, output_list,pred,True,True)

#output_list = []
while True :
    #img = get_hand_write()
    gochange = False
    if is_change :
        output_list = []
        pred = an(torch.Tensor(img.reshape(1,1,28,28),device=device)/255.0).argmax()
        #print(output_list[-1])
        #last_time = time.time()
    elif auto_show and (time.time() - last_time > wait_time) :
        #print(time.time())
        output_list = []
        img = test_img[random.randint(0,1000)].reshape(28,28)
        pred = an(torch.Tensor(img.reshape(1,1,28,28),device=device)/255.0).argmax()
        last_time = time.time()
        gochange = True
    elif not auto_show:
        last_time = 0


    img, is_change,auto_show = update_screen(screen, img, output_list,pred,auto_show,gochange)
    #show_output()

