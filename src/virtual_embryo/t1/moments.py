"""Population moments with the original row reduction order."""
import numpy as np

def moments(x, block_rows=128):
    n,g=x.shape
    total=np.zeros(g,np.float64); second=np.zeros(g,np.float64)
    for lo in range(0,n,block_rows):
        a=np.asarray(x[lo:lo+block_rows],dtype=np.float64)
        total+=a.sum(0);second+=np.square(a).sum(0)
    mu=total/n
    return mu,np.maximum(second/n-mu*mu,0)
