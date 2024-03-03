import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from math import sqrt

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def f(x, y):
    #return torch.sin(x)/x  * y#torch.sin(y)
    return torch.sin(torch.sqrt(x**2+y**2)/2)/5
    #return x*0.5+y*2
    #return (1 - x/2 + x**5 + y**3) * torch.exp(-x**2 - y**2)



lr = 0.2
alpha = 0.2
beta = 0.96
xs = 0.5
ys = 0

show_range = 20
steps = 50
x = torch.tensor(-0.15)
x.requires_grad_(True)
y = torch.tensor(-0.15)
y.requires_grad_(True)

history = [[],[],[]]

x_r1,x_r2 = x.item()-show_range, x.item()+show_range
y_r1,y_r2 = y.item()-show_range, y.item()+show_range
xx = torch.linspace(x_r1,x_r2, steps)
yy = torch.linspace(y_r1,y_r2, steps)
X, Y = torch.meshgrid(xx, yy, indexing='ij')
Z = f(X, Y)

def animate(i):
    global x, y, history,X,Y,Z,x_r1,x_r2,y_r1,y_r2,xs,ys
    print(x.item(),y.item())
    print(xs,ys)
    z = f(x,y)
    z.backward()
    
    history[0].append(x.detach())
    history[1].append(y.detach())
    history[2].append(z.detach())
    #x.data = x.data - lr * x.grad
    #y.data = y.data - lr * y.grad
    xi,yi = x.grad.item(),y.grad.item()
    
    xs *= beta 
    ys *= beta
    
    xs -= lr * xi/sqrt(1+xi**2)
    ys -= lr * yi/sqrt(1+yi**2)


    x.data = x.data + xs
    y.data = y.data + ys
    

    x.grad.data.zero_()
    y.grad.data.zero_()
    


    
    #x_r1,x_r2 = x_last-show_range, x_last+show_range
    #y_r1,y_r2 = y_last-show_range, y_last+show_range
    if (x.item() < x_r1 + show_range/2) or  (x_r2 - show_range/2 < x.item()) or (y.item() < y_r1 + show_range/2) or  (y_r2 - show_range/2 < y.item()) :
        x_r1,x_r2 = x.item()-show_range, x.item()+show_range
        y_r1,y_r2 = y.item()-show_range, y.item()+show_range
        xx = torch.linspace(x_r1,x_r2, steps)
        yy = torch.linspace(y_r1,y_r2, steps)
        X, Y = torch.meshgrid(xx, yy, indexing='ij')
        Z = f(X, Y)


        
    ax.clear()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    #ax.scatter(history[0],history[1],history[2],c="red",s=20)
    ax.plot(history[0][-10:],history[1][-10:],history[2][-10:],c="red",linestyle='-',marker='o', zorder=100)
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


ani = animation.FuncAnimation(fig, animate, frames=50, interval=800)
plt.show()
