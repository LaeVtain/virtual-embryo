"""Fixed 32-step conditional expression generation."""
import numpy as np
import torch

@torch.inference_mode()
def latent(e,x,start,end,times,seed=101,batch=4096):
    # Generate the same row-major noise before batching.
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed))
    out=np.empty_like(x)
    for a in range(0,len(x),batch):
        xx=torch.from_numpy(x[a:a+batch]);state=noise[a:a+batch].clone()
        for k in range(32):
            u=xx.new_full((len(xx),),k/32)
            state+=e(state,u,xx,u*0+start,u*0+(end-start),times[0],times[2])/32
        out[a:a+batch]=state.numpy()
    return out


def training_rollout(net, source, start, end, tmin, tmax, noise, steps=8):
    state=noise
    for i in range(steps):
        u=source.new_full((len(source),),i/steps)
        state=state+net(state,u,source,u*0+start,u*0+(end-start),tmin,tmax)/steps
    return state.relu()
