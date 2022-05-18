from distutils.log import error
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from fastai.torch_core import *

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    try:
        conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight)
        if bias: conv.bias.data.zero_()
        return torch.nn.utils.spectral_norm(conv)
    except:
        print("error")
    # conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    # nn.init.kaiming_normal_(conv.weight)
    # if bias: conv.bias.data.zero_()
    # return torch.nn.utils.spectral_norm(conv)

def MLP(channels: list, do_bn=True):
    '''
    Multi-layer perceptron
    :param channels:
    :param do_bn:
    :return:
    '''
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
                layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class PointEncoder(nn.Module):

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, pts):
        return self.encoder(pts.transpose(1, 0).contiguous().unsqueeze(0))
        

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

class CrossAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class SelfAttention(nn.Module):

  "Self attention layer for nd."

  def __init__(self, n_channels:int):
      super().__init__()
      self.query = conv1d(n_channels, n_channels)
      self.key   = conv1d(n_channels, n_channels)
      self.value = conv1d(n_channels, n_channels)
      self.gamma = nn.Parameter(torch.as_tensor([0.]))

  def forward(self, x):
      # Notation from https://arxiv.org/pdf/1805.08318.pdf
      size = x.size()
      x = x.view(*size[:2],-1)
      f,g,h = self.query(x),self.key(x),self.value(x)
      beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
      o = self.gamma * torch.bmm(h, beta) + x
      
      return o.view(*size).contiguous()

class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1, sym=True):#, n_out:int):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)      
        self.gamma = nn.Parameter(tensor([0.]))
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)
                
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2) 
        o = self.gamma * o + x
        
        return o.view(*size).contiguous()