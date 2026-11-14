"""Shared training and inference networks with original parameter names."""
import torch
from torch import nn

class ConditionalEndpoint(nn.Module):
    def __init__(self,n_genes,hidden=128,nblock=2):
        super().__init__()
        self.inp=nn.Linear(2*n_genes+3,hidden)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(nblock)])
        self.act=nn.SiLU()
        self.out=nn.Linear(hidden,n_genes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self,state,s,source,t_start,duration,tmin,tmax):
        span=float(tmax)-float(tmin)
        clocks=torch.stack((s,(t_start-tmin)/span,duration/span),dim=1)
        h=self.act(self.inp(torch.cat((state,source,clocks),dim=1)))
        for block in self.blocks:h=h+block(h)
        return self.out(h)

class ContinuousNeighborhoodField(nn.Module):
    def __init__(self,feature_dimension,hidden=32,feature_hidden=16,k=15):
        super().__init__()
        self.feature_dimension,self.k=feature_dimension,k
        self.feature_encoder=nn.Sequential(nn.Linear(feature_dimension,feature_hidden),nn.Tanh())
        self.radial=nn.Sequential(nn.Linear(feature_hidden+2,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.edge=nn.Sequential(nn.Linear(2*feature_hidden+4,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        for layer in (self.radial[-1],self.edge[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, positions, features):
        from .spatial import dense_field
        return dense_field(self, positions, features)
