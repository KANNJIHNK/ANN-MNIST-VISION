import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from math import sqrt

fig = plt.figure(figsize=(16,8))
#fig.subplots(1,2)

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax2 = fig.add_subplot(1, 2, 2)
#ax2.set_aspect('equal')

def f(x, y,d4):
    #!return torch.sin(x)/x  + (torch.sin(y)/y * d4)**2
    return torch.sin(torch.sqrt(x**2+y**2+d4**2)/2)/5
    #return x*0.5+y*2
    #return (1 - x/2 + x**5 + y**3 + 3*d4**2 + d4*2) * torch.exp(-x**2 - y**2 - d4**2)



lr = 0.8
beta = 0.97
xs = -0.2
ys = 0.2
d4s = 0.8

show_range = 15
steps = 50
x = torch.tensor(-3.15)
x.requires_grad_(True)
y = torch.tensor(-1.15)
y.requires_grad_(True)
d4 = torch.tensor(-1.0)
d4.requires_grad_(True)

#history = [[],[],[]]

x_r1,x_r2 = x.item()-show_range, x.item()+show_range
y_r1,y_r2 = y.item()-show_range, y.item()+show_range
xx = torch.linspace(x_r1,x_r2, steps)
yy = torch.linspace(y_r1,y_r2, steps)
X, Y = torch.meshgrid(xx, yy, indexing='ij')
Z = f(X, Y,d4.data)

def animate(i):
    global x, y, history,X,Y,Z,d4,x_r1,x_r2,y_r1,y_r2,xs,ys,d4s
    if i%10 == 0 :
        print(x.item(),y.item(),d4.item())
    #print(xs,ys)
    z = f(x,y,d4)
    z.backward()
    
    #history[0].append(x.detach())
    #history[1].append(y.detach())
    #history[2].append(z.detach())
    #x.data = x.data - lr * x.grad
    #y.data = y.data - lr * y.grad

    xi,yi,d4i = x.grad.item(),y.grad.item(),d4.grad.item()

    ax.clear()
    ax2.clear()

    #ax2.set_aspect('equal')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    ax.plot([x.item()],[y.item()],[z.item()],c="red",linestyle='-',marker='o', zorder=100)

    ad4 = torch.linspace(d4.item()-show_range,d4.item()+show_range, steps*3)
    
    aF = f(x.data,y.data,ad4.data)
    bd4 = f(x.data,y.data,d4.data)
    
    ax2.plot(ad4,aF,c="blue",linestyle='-',marker='')
    ax2.plot([d4.item()],[bd4],c="red",linestyle='-',marker='o', zorder=100)
    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    ax2.set_xlabel('d4')
    ax2.set_ylabel('z')
    
    
    xs *= beta 
    ys *= beta
    d4s *= beta
    
    xs -= lr * xi/sqrt(1+xi**2)
    ys -= lr * yi/sqrt(1+yi**2)
    d4s -= lr * d4i/sqrt(1+d4i**2)


    x.data = x.data + xs
    y.data = y.data + ys
    d4.data = d4.data + d4s
    

    x.grad.data.zero_()
    y.grad.data.zero_()
    d4.grad.data.zero_()
    
    if (x.item() < x_r1 + show_range/2) or  (x_r2 - show_range/2 < x.item()) or (y.item() < y_r1 + show_range/2) or  (y_r2 - show_range/2 < y.item()) :    
        x_r1,x_r2 = x.item()-show_range, x.item()+show_range
        y_r1,y_r2 = y.item()-show_range, y.item()+show_range
        xx = torch.linspace(x_r1,x_r2, steps)
        yy = torch.linspace(y_r1,y_r2, steps)
        X, Y = torch.meshgrid(xx, yy, indexing='ij')
    Z = f(X, Y,d4.data)


        
   #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='viridis')



ani = animation.FuncAnimation(fig, animate, frames=50, interval=100)
plt.show()
