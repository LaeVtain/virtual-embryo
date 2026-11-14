"""Support-only global projection with analytic softmax gradients.

The support is fixed by the frozen candidate. Missing entries have exactly zero
output and contribute no logit derivative. Global moments and the Adam schedule
are unchanged; only intermediate representation and floating-point reductions differ.
"""
import time
import numpy as np
import torch
from .moments import moments


def project(original,fixed,weight,variance_reference=None,block_rows=128,
            output_path=None,progress=None):
    began=time.monotonic();n,g=original.shape
    desired=np.asarray(fixed,np.float64) if np.ndim(fixed)==1 else moments(fixed,block_rows)[0]
    base_mean,base_var=moments(original,block_rows)
    if variance_reference is None:v0=base_var
    elif isinstance(variance_reference,tuple):v0=variance_reference[1]
    else:v0=moments(variance_reference,block_rows)[1]
    mean_scale=max(float(np.square(base_mean-desired).mean()),1e-12)
    var_scale=max(float(np.square(v0).mean()),1e-12)
    blocks=[];counts=np.zeros(g,np.int64);center_sum=np.zeros(g,np.float64)
    for lo in range(0,n,block_rows):
        a=np.asarray(original[lo:lo+block_rows],np.float64)
        row,col=np.nonzero(a>0);nr=len(a)
        rc=np.bincount(row,minlength=nr)
        ptr=np.r_[0,np.cumsum(rc)]
        b=np.log(np.expm1(a[row,col]))
        center_sum+=np.bincount(col,weights=b,minlength=g)
        counts+=np.bincount(col,minlength=g)
        blocks.append((lo,nr,row.astype(np.int32),col.astype(np.int32),ptr,b))
    center=center_sum/np.maximum(counts,1)
    # Cache the per-entry centered log intensity once.
    blocks=[(*block,block[5]-center[block[3]]) for block in blocks]
    bias=torch.zeros(g,dtype=torch.float64,requires_grad=True)
    temperature=torch.zeros_like(bias,requires_grad=True)
    opt=torch.optim.Adam([bias,temperature],lr=.03)
    def forward(block,b,te):
        _,_,rows,cols,ptr,base,centered=block
        logits=base+b[cols]+(te[cols]-1)*centered
        maximum=np.maximum.reduceat(logits,ptr[:-1])
        e=np.exp(logits-maximum[rows]);s=np.add.reduceat(e,ptr[:-1])
        q=e/s[rows]
        y=np.log1p(10000*q)
        return y,q
    logs=[];loop_began=time.monotonic()
    for step in range(300):
        b=bias.detach().numpy();te=np.exp(temperature.detach().numpy())
        sy=np.zeros(g);sy2=np.zeros(g)
        for block in blocks:
            y,_=forward(block,b,te);col=block[3]
            sy+=np.bincount(col,weights=y,minlength=g)
            sy2+=np.bincount(col,weights=y*y,minlength=g)
        mu=sy/n;var=sy2/n-mu*mu
        m=float(np.square(mu-desired).mean()/mean_scale)
        v=float(np.square(var-v0).mean()/var_scale)
        dm=2*(mu-desired)/(g*mean_scale*n)
        dv=4*weight*(var-v0)/(g*var_scale*n)
        gb=np.zeros(g);gt=np.zeros(g)
        for block in blocks:
            _,_,rows,col,ptr,base,centered=block
            y,q=forward(block,b,te)
            dy=dm[col]+dv[col]*(y-mu[col])
            r=dy*(10000*q)/(1+10000*q)
            # J_softmax^T grad for log1p(10000*q), within each whole row.
            dl=r-q*np.add.reduceat(r,ptr[:-1])[rows]
            gb+=np.bincount(col,weights=dl,minlength=g)
            gt+=np.bincount(col,weights=dl*centered,minlength=g)
        opt.zero_grad(set_to_none=True)
        bias.grad=torch.from_numpy(gb)
        temperature.grad=torch.from_numpy(gt*te)
        opt.step()
        if step%50==0:
            item=dict(step=step,mean_loss=m,variance_loss=v);logs.append(item)
            if progress:progress(item)
    output=(np.lib.format.open_memmap(output_path,mode='w+',dtype=np.float64,shape=(n,g))
            if output_path is not None else np.empty((n,g),np.float64))
    b=bias.detach().numpy();te=np.exp(temperature.detach().numpy())
    for block in blocks:
        lo,nr,rows,col,*_=block;y,_=forward(block,b,te)
        a=np.zeros((nr,g),np.float64);a[rows,col]=y
        output[lo:lo+nr]=a
    if isinstance(output,np.memmap):output.flush()
    return output,dict(logs=logs,seconds=time.monotonic()-began,
        optimizer_seconds=time.monotonic()-loop_began,
        implementation='global_sparse_support_analytic_gradient',block_rows=block_rows,
        support_nnz=int(counts.sum()),support_density=float(counts.sum()/(n*g)))
