"""Disk-backed frozen T1 inference; row blocks retain population-wide constraints."""
import time
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import open_memmap
from .model_loader import load_model
from .projection import project

def noise_blocks(rows,genes,latent_dim,seed,batch=32):
    """Identical independent NumPy streams, consumed in C row order."""
    names=['latent_normal','mask_first_uniform','mask_uniform','positive_normal']
    rngs=[np.random.default_rng(np.random.SeedSequence([seed,i,9711])) for i in range(4)]
    for lo in range(0,rows,batch):
        n=min(batch,rows-lo);shapes=[(n,latent_dim),(n,),(n,genes),(n,genes)]
        yield lo,{name:torch.from_numpy(rng.standard_normal(shape) if name.endswith('normal') else rng.random(shape)) for name,rng,shape in zip(names,rngs,shapes)}

def allocate(path,shape,dtype='float64'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    return open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def dense_batch(x,lo,n):
    b=x[lo:lo+n]
    if hasattr(b,'toarray'):b=b.toarray()
    return torch.from_numpy(np.asarray(b,dtype=np.float32))

@torch.inference_mode()
def sample(net,x,start,end,seed,path,progress=None):
    out=allocate(path,(x.shape[0],net.n_genes));began=time.monotonic()
    for lo,r in noise_blocks(len(x) if not hasattr(x,'tocsr') else x.shape[0],net.n_genes,net.latent_dim,seed):
        n=len(r['latent_normal']);y=net.sample_prior(dense_batch(x,lo,n),r,start,end,output_dtype=torch.float64)
        out[lo:lo+n]=y.numpy()
        if progress and lo%1024==0:progress('sample',lo,out.shape[0])
    out.flush();return out,dict(seconds=time.monotonic()-began)

def column_mean(a):
    # NumPy reduces axis 0 row-sequentially without constructing a matrix copy.
    return np.asarray(a).mean(0)

@torch.inference_mode()
def teacher_mean(model,x,seed,path,progress=None):
    rows=x.shape[0];out=allocate(path,(rows,model.n_genes))
    nets=[model.model(f'teacher{s}') for s in model.manifest['teacher_seeds']]
    streams=[noise_blocks(rows,net.n_genes,net.latent_dim,seed+offset) for net,offset in zip(nets,model.manifest['noise_offsets'])]
    for lo in range(0,rows,32):
        xx=dense_batch(x,lo,min(32,rows-lo));parts=[]
        for net,stream in zip(nets,streams):
            _,r=next(stream)
            y=net.sample_prior(xx,r,9.5,10.5,output_dtype=torch.float64);parts.append(y.numpy())
        out[lo:lo+len(xx)]=np.mean(parts,axis=0)
        if progress and lo%1024==0:progress('teachers',lo,rows)
    out.flush();return column_mean(out)

@torch.inference_mode()
def structure(original,x,net,seed,directory,progress=None,gene_block=128):
    d=Path(directory);rows,genes=original.shape
    logits=allocate(d/'logits.npy',(rows,genes));scores=allocate(d/'scores.npy',(rows,genes))
    for lo,r in noise_blocks(rows,genes,net.latent_dim,seed):
        xx=dense_batch(x,lo,len(r['latent_normal']));h=net._condition(xx,9.5,10.5);q=net._gaussian(net.prior(h))
        z=q['mean'].double()+(.5*q['logvar'].double()).exp()*r['latent_normal'];p=net._decode(h,z)
        logits[lo:lo+len(xx)]=(p['location'].double()+p['scale'].double()*r['positive_normal']).numpy()
        scores[lo:lo+len(xx)]=(p['gate_logits'].double()-torch.logit(r['mask_uniform'])).numpy()
    mask=allocate(d/'mask.npy',(rows,genes),'bool');counts=np.zeros(genes,dtype=np.int64)
    for lo in range(0,rows,128):counts+=(original[lo:lo+128]>0).sum(0)
    for g in range(0,genes,gene_block):
        hi=min(genes,g+gene_block);order=np.argsort(scores[:,g:hi],axis=0,kind='stable')
        ranks=np.empty_like(order);np.put_along_axis(ranks,order,np.arange(rows)[:,None],axis=0)
        mask[:,g:hi]=ranks>=rows-counts[g:hi]
        if progress and g%4096==0:progress('rank_genes',g,genes)
    out=allocate(d/'base.npy',(rows,genes))
    for lo in range(0,rows,32):
        m=torch.from_numpy(np.asarray(mask[lo:lo+32]))
        b=torch.from_numpy(np.asarray(logits[lo:lo+32]));y=torch.log1p(10000*torch.softmax(torch.where(m,b,-torch.inf),dim=1)).numpy()
        out[lo:lo+len(y)]=y
    out.flush();mask.flush();logits.flush();scores.flush();return out

def predict(model,x,start,end,seed,directory,progress=None,projection_block=128):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    if (start,end)!=(9.5,10.5):
        y,a=sample(model.model('fullELBO30'),x,start,end,seed,d/'candidate.npy',progress)
        a.update(route='fullELBO30_unchanged',seconds=time.monotonic()-began);return y,a
    original,oa=sample(model.model('original'),x,start,end,seed,d/'original.npy',progress)
    desired=teacher_mean(model,x,seed,d/'teacher_average.npy',progress)
    base=structure(original,x,model.model('fullELBO30'),seed,d,progress)
    reference_scale=max(float(np.square(column_mean(original)-desired).mean()),1e-12)
    base_scale=max(float(np.square(column_mean(base)-desired).mean()),1e-12)
    weight=100*reference_scale/base_scale
    callback=(lambda item:progress('projection',item['step'],300)) if progress else None
    with torch.enable_grad():y,a=project(base,desired,weight,variance_reference=original,block_rows=projection_block,output_path=d/'candidate.npy',progress=callback)
    a.update(route='original_gene_counts_fullELBO_joint_structure_independent_mean3',equivalent_variance_weight=weight,original_sample=oa,seconds=time.monotonic()-began)
    return y,a
