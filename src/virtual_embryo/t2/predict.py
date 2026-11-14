"""T2 source-only prediction with frozen marginal maps and sparse spatial field."""
from pathlib import Path
import json,time,argparse
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from ..paths import task_assets
from .networks import ConditionalEndpoint, ContinuousNeighborhoodField
from .spatial import rollout,landmark_rollout,spatial_features
from .expression import latent

def quantile_map(z,knots,alpha=1.):
    p,t=knots;out=np.empty_like(z)
    for g in range(z.shape[1]):
        v=z[:,g];mapped=np.interp(v,p[:,g],t[:,g])
        out[:,g]=np.maximum(0,(1-alpha)*v+alpha*mapped)
    return out

def mean_preserving_map(z,knots):
    q=quantile_map(z,knots)
    original=np.maximum(z,0)
    base=original.mean(0,dtype=np.float64);qm=q.mean(0,dtype=np.float64);active=qm>0
    q[:,active]*=(base[active]/qm[active]).astype('float32')
    q[:,~active]=original[:,~active]
    upper=np.maximum(original.max(0),knots[1][-1])
    for g in np.flatnonzero(q.max(0)>upper+1e-6):
        positive=q[:,g]>0
        values=np.minimum(q[positive,g].astype(np.float64),upper[g])
        deficit=base[g]*len(q)-values.sum()
        capacity=float(upper[g])*len(values)-values.sum()
        if deficit>capacity:
            q[:,g]=original[:,g]
        else:
            if capacity>0:values+=(upper[g]-values)*(max(0.,deficit)/capacity)
            q[positive,g]=values
    return q

class Predictor:
    def __init__(self,setting,baseline=False,model_dir=None):
        folder=Path(model_dir) if model_dir is not None else task_assets("t2")
        self.setting=setting;self.baseline=baseline
        self.config=json.loads((folder/'config.json').read_text())
        self.times=self.config[setting]['times'];self.unit=self.config[setting]['coordinate_unit']
        md=json.loads((folder/'metadata.json').read_text())[f'T2_{setting}'];self.genes=md['genes'];g=len(self.genes)
        ck=torch.load(folder/f'{setting}_control.pt',map_location='cpu',weights_only=True)
        self.expression=ConditionalEndpoint(g,hidden=128,nblock=2).eval().requires_grad_(False);self.expression.load_state_dict(ck['expression'])
        self.spatial=ContinuousNeighborhoodField(g+2,hidden=32,feature_hidden=16,k=15).eval().requires_grad_(False)
        if setting=='heart' and not baseline:
            ck=torch.load(folder/'heart_spatial_continued.pt',map_location='cpu',weights_only=True)
        self.spatial.load_state_dict(ck['spatial'])
        self.maps=None if baseline else joblib.load(folder/f'{setting}_maps.joblib')

    @torch.inference_mode()
    def predict(self,x,c,*,genes,source_time,target_time,seed=101,batch_size=4096):
        supplied=np.asarray(genes).astype(str).tolist()
        if len(supplied)!=np.shape(x)[1] or len(set(supplied))!=len(supplied):
            raise ValueError("Input genes must uniquely name every expression column")
        columns={gene:i for i,gene in enumerate(supplied)}
        missing=set(self.genes)-columns.keys()
        if missing:
            raise ValueError(f"Missing required genes: {sorted(missing)}")
        x=np.asarray(x,dtype='float32')
        if supplied!=self.genes:
            x=np.ascontiguousarray(x[:,[columns[gene] for gene in self.genes]])
        c=np.asarray(c,dtype='float32');center=c.mean(0)
        start=time.perf_counter()
        expression_latent=latent(self.expression,x,source_time,target_time,self.times,seed,batch_size)
        expression_seconds=time.perf_counter()-start;tick=time.perf_counter()
        if self.baseline:px=np.maximum(expression_latent,0)
        elif target_time<=self.times[2]:
            pair=(self.times.index(source_time),self.times.index(target_time));px=quantile_map(expression_latent,self.maps[pair])
        elif self.setting=='heart':px=mean_preserving_map(expression_latent,self.maps[1,2])
        else:px=quantile_map(expression_latent,self.maps[0,2],alpha=.5)
        calibration_seconds=time.perf_counter()-tick;del expression_latent
        tick=time.perf_counter();xx=torch.from_numpy(x);cc=torch.from_numpy((c-center)/self.unit)
        f=spatial_features(xx,source_time,target_time,self.times[0],self.times[2]-self.times[0])
        if self.setting=='heart' and not self.baseline:
            sc=self.config['heart']['spatial']
            pc=landmark_rollout(self.spatial,cc,f,target_time-source_time,anchor_count=sc['anchor_count'],anchor_k=sc['anchor_k'],seed=sc['seed']).numpy()
        else:pc=rollout(self.spatial,cc,f,target_time-source_time).numpy()
        spatial_seconds=time.perf_counter()-tick
        return dict(x=px,c=pc.astype('float64')*self.unit+center,c_centered_unit=pc,genes=np.asarray(self.genes),seconds=dict(expression=expression_seconds,calibration=calibration_seconds,spatial=spatial_seconds,total=time.perf_counter()-start))

def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',choices=['heart','embryo'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-time',type=float,required=True);p.add_argument('--target-time',type=float,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--threads',type=int,default=1);p.add_argument('--baseline',action='store_true');p.add_argument('--model-dir',type=Path);a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):
        model=Predictor(a.setting,baseline=a.baseline,model_dir=a.model_dir)
        with np.load(a.input,allow_pickle=False) as z:out=model.predict(z['x'],z['c'],genes=z['genes'],source_time=a.source_time,target_time=a.target_time,seed=a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**{k:v for k,v in out.items() if k!='seconds'})
    a.output.with_suffix('.json').write_text(json.dumps(out['seconds'],indent=2));print(json.dumps(out['seconds']))

if __name__=='__main__':main()
