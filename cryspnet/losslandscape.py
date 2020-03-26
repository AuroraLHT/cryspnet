import torch
from fastai.torch_core import *
import copy


__all__ = ['LossLandscape', 'plot_landscape', 'plot_landscape_contour']

def randomize( linear ):
    weight = linear.weight
    bias = linear.bias
    
    wn = weight.norm(dim=1, keepdim=True)
#     bn = bias.norm(dim=1, keepdim=True)
    
    randn_w = torch.randn_like( weight )
#     randn_b = torch.randn_like( bias )
    
    randwn = randn_w.norm(dim=1, keepdim=True)
#     randbn = randn_b.norm(dim=1, keepdim=True)
    
    randn_w = randn_w / (randwn / wn)
#     randn_b = randn_b  / bias.data)
    
    weight.data.copy_(randn_w)
#     bias.data.copy_(randn_b)

def randomize_model(m):    
    for child in children(m):
        if isinstance( child, nn. Linear ):
            randomize( child )
        else:
            randomize_model( child )
    return m

def linear_morph1d(m_fin, m_ran1, m_out, x1):
    childrens_o, childrens_f, childrens_r1, childrens_r2 = children(m_out)[3], children(m_fin)[3], children(m_ran1)[3], children(m_ran2)[3]
    
    for child_o, child_f, child_r1 in zip(childrens_o, childrens_f, childrens_r1, childrens_r2):
        if isinstance(child, nn.Linear):
            w_out,w_fin, w_ran1 = child_o.weight, child_f.weight, child_r1.weight
            b_out,b_fin, b_ran1 = child_o.bias, child_f.bias, child_r1.bias
            
            w_out = w_ran1 * x1 + w_ran2
            b_out = b_ran1 * x1 + b_ran2
        
    return m_out

def linear_morph2d(m_fin, m_ran1, m_ran2, m_out, x1, x2):
    childrens_o, childrens_f, childrens_r1, childrens_r2 = children(m_out)[3], children(m_fin)[3], children(m_ran1)[3], children(m_ran2)[3]
    
    for child_o, child_f, child_r1, child_r2 in zip(childrens_o, childrens_f, childrens_r1, childrens_r2):
        if isinstance(child_o, nn.Linear):
            w_out,w_fin, w_ran1, w_ran2 = child_o.weight, child_f.weight, child_r1.weight, child_r2.weight
#             b_out,b_fin, b_ran1, b_ran2 = child_o.bias, child_f.bias, child_r1.bias, child_r2.bias
            
            w_out.data.copy_( w_ran1.data * x1 + w_ran2.data * x2 + w_fin.data )
#             b_out.data.copy_( b_ran1.data * x1 + b_ran2.data * x2 + b_fin.data )
    return m_out


class LossLandscape:
    def __init__(self, learn):
        self.learn = learn
        self.m = learn.model
    
    def prob2D(self, start=-1, end=1, n=10):
        val_err = np.zeros((n,n,))
        mo = copy.deepcopy(self.m)
        mr1 = randomize_model( copy.deepcopy(self.m) ) 
        mr2 = randomize_model( copy.deepcopy(self.m) )

        for i, x1 in enumerate(np.linspace(start, end, n)):
            for j, x2 in enumerate(np.linspace(start, end, n)):
                mo = linear_morph2d(self.m, mr1, mr2, mo, x1, x2)
                self.learn.model = mo
                val_err[i, j] = self.learn.validate()[0]
        
        self.reset()
        return val_err
    
    def prob1D(self, start=-1, end=1, n=10):
        val_err = np.zeros(n)
        mo = copy.deepcopy(self.m)
        mr1 = randomize_model( copy.deepcopy(self.m) )

        for i, x1 in enumerate(np.linspace(start, end, n)):
            mo = linear_morph1d(self.m, mr1, mo, x1, x2)
            self.learn.model = mo
            val_err[i] = self.learn.validate()[0]
        
        self.reset()
        return val_err
    
    def reset(self):
        self.learn.model = self.m
        
        
def plot_landscape(landscape, start=-1, end=1, n=10, figsize=(9,9)):
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.imshow(landscape, cmap='gray')
#     plt.yticks( np.arange(n), np.linspace(start, end, n) )
#     plt.xticks( np.arange(n), np.linspace(start, end, n), rotation=45, ha = 'right' )

    plt.colorbar()
    
    return fig, ax
    
    
def plot_landscape_contour(landscape, start=-1, end=1, n=10, figsize=(9,9), lmin=None, lmax=None, density=10):
    fig, ax = plt.subplots(figsize=figsize)
    
    level = np.linspace(lmin, lmax, density)
    
    X, Y= np.meshgrid(np.linspace(start, end, n), np.linspace(start, end, n))
    CS=plt.contour(X, Y, landscape, level)
    ax.clabel(CS, inline=1, fontsize=10)
    
    return fig, ax