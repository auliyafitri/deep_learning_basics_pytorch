import matplotlib.pyplot as plt
import numpy as np
import torch 

def print_function(func, name="", start=-3, end=3, color="black", title="", ax=None):

    x = torch.arange(start=start, end=end, step=0.01)

    if ax is None:
        plt.plot(x, func(x).numpy(), color=color, label=name)
        plt.title(title, fontsize=18)
        if name!="":
            plt.legend(fontsize=15)
    else:
        ax.plot(x, func(x).numpy(), color=color, label=name)
        ax.set_title(title, fontsize=18)
        if name!="":
            ax.legend(fontsize=15)

def print_classification_regions(X_val, Y_val, f=None, ax=None, title=""):

    # Plot Training Data
    x = torch.linspace(-8., 8., 1000)
    y = torch.linspace(-8., 8., 1000)
    [X, Y] = torch.meshgrid(x, y)
    
    if f is not None: 
        ax.contourf(X,Y, (torch.tanh(f(torch.stack((X,Y),1)).squeeze())).float(), levels=[-10, 0, 10])

    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)

    ax.scatter(X_val[Y_val[:,0]==0,0], X_val[Y_val[:,0]==0,1], color="blue", label="Class 1")
    ax.scatter(X_val[Y_val[:,0]==1,0], X_val[Y_val[:,0]==1,1], color="green", label="Class 2")
    ax.set_title(title, fontsize=15)
    ax.legend(loc='upper right', fontsize=15)
    ax.axis('equal');



