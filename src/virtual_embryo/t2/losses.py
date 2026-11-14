"""T2 training-only metric surrogates with frozen training statistics."""
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import rankdata

WEIGHTS=dict(direction=.125,mmd=.15,variogram=.10,shape=1/12,scale=1/12,neighborhood=.25)

def squared_distances(a,b):
    return (a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)


def multiscale_mmd(a,b,gamma):
    if min(len(a),len(b))<2:raise ValueError('Unbiased MMD requires >=2 cells')
    aa,bb,ab=squared_distances(a,a),squared_distances(b,b),squared_distances(a,b)
    ia=1-torch.eye(len(a),device=a.device,dtype=a.dtype)
    ib=1-torch.eye(len(b),device=b.device,dtype=b.dtype)
    terms=[]
    for scale in (.25,.5,1.,2.,4.):
        terms.append((torch.exp(-gamma*scale*aa)*ia).sum()/(len(a)*(len(a)-1))+
                     (torch.exp(-gamma*scale*bb)*ib).sum()/(len(b)*(len(b)-1))-
                     2*torch.exp(-gamma*scale*ab).mean())
    return torch.stack(terms).mean()


def partial_direction(pred_delta,reference_rank,target_residual,landmarks,tau):
    # Monte Carlo soft ranks over fixed gene landmarks, without a G x G matrix.
    ranks=torch.sigmoid((pred_delta[:,None]-pred_delta[landmarks][None,:])/tau).mean(1)
    ranks=ranks-ranks.mean();rr=reference_rank-reference_rank.mean()
    residual=ranks-(ranks@rr)/(rr@rr).clamp_min(1e-8)*rr
    return (residual@target_residual)/(residual.norm()*target_residual.norm()).clamp_min(1e-8)


def tensors(stats, device):
    return {k: torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v for k, v in stats.items()}

def _rms(c):
    return (c - c.mean(0)).square().sum(1).mean().clamp_min(1e-12).sqrt()

def soft_neighborhood(x, c, bandwidth):
    """Smooth local averaging with self and a fixed training bandwidth."""
    return torch.softmax(-squared_distances(c, c) / bandwidth, dim=1) @ x

def _project_fit(x, seed):
    p = PCA(n_components=min(30, x.shape[1], len(x) - 1), svd_solver='randomized', random_state=seed).fit(x)
    z = p.transform(x).astype('float32')
    d = ((z[:, None] - z[None, :]) ** 2).sum(-1)
    return (p.mean_.astype('float32'), p.components_.astype('float32'), z, 1 / max(float(np.median(d[d > 0])) if (d > 0).any() else 1.0, 1e-08))

def raw_terms(pred, s, coords):
    """Expression and coordinate surrogate terms for generated cells."""
    w = pred.new_full((len(pred),), 1 / len(pred))
    mean = w @ pred
    delta = mean - s['reference_mean']
    terms = {'direction': 1 - partial_direction(delta, s['reference_rank'], s['target_residual'], s['landmarks'], s['temperature'])}
    z = (pred - s['pca_mean']) @ s['pca_components'].T
    terms['mmd'] = multiscale_mmd(z, s['target_projection'], s['gamma'])
    v = w @ (torch.sqrt((pred[:, s['gene_i']] - pred[:, s['gene_j']]).abs() + 1e-06) - 0.001)
    terms['variogram'] = (v - s['target_variogram']).square().mean()
    if coords is None:
        raise ValueError('T2 needs generated coordinates and unweighted cells')
    rms = _rms(coords)
    q = torch.linspace(0, 1, 128, device=coords.device)
    distances = torch.pdist(coords / rms)
    terms['shape'] = (torch.quantile(distances, q) - s['shape_quantiles']).abs().mean()
    terms['scale'] = torch.log(rms / s['target_rms']).square()
    local = soft_neighborhood(pred, coords, s['neighbor_bandwidth'])
    local = (local - s['neighbor_mean']) @ s['neighbor_components'].T
    terms['neighborhood'] = multiscale_mmd(local, s['neighbor_projection'], s['neighbor_gamma'])
    if not all((torch.isfinite(v) for v in terms.values())):
        raise ValueError('Nonfinite metric surrogate')
    return terms

def fit_statistics(reference, target, *, reference_coords, target_coords, seed=20261102):
    """Fit training projections and discrepancy scales; direction keeps scale one."""
    r, y = (np.asarray(reference, dtype='float32'), np.asarray(target, dtype='float32'))
    if min(len(r), len(y)) < 64 or r.shape[1] != y.shape[1]:
        raise ValueError('At least 64 aligned training rows needed')
    rng = np.random.default_rng(seed)
    g = y.shape[1]
    rm, tm = (r.mean(0), y.mean(0))
    rr = rankdata(rm).astype('float32') / g
    rr -= rr.mean()
    tr = rankdata(tm - rm).astype('float32') / g
    tr -= tr.mean()
    tr -= tr.dot(rr) / max(rr.dot(rr), 1e-08) * rr
    pm, pc, yz, gamma = _project_fit(y[:256], seed)
    gi = rng.integers(g, size=2048)
    gj = rng.integers(g, size=2048)
    keep = gi != gj
    gi, gj = (gi[keep], gj[keep])
    s = dict(reference_mean=rm, target_mean=tm, reference_rank=rr, target_residual=tr, landmarks=np.sort(rng.choice(g, min(128, g), replace=False)), temperature=max(0.01, 0.1 * float(np.std(tm - rm))), pca_mean=pm, pca_components=pc, target_projection=yz, gamma=gamma, gene_i=gi, gene_j=gj, target_variogram=(np.sqrt(np.abs(y[:, gi] - y[:, gj]) + 1e-06) - 0.001).mean(0))
    c = torch.as_tensor(np.asarray(target_coords[:256], dtype='float32'))
    xx = torch.from_numpy(y[:256])
    rms = _rms(c)
    d = squared_distances(c, c)
    bw = float(torch.median(d.topk(min(15, len(c)), largest=False).values[:, -1]).clamp_min(1e-06))
    local = soft_neighborhood(xx, c, bw).numpy()
    nm, nc, nz, ng = _project_fit(local, seed)
    s.update(target_rms=float(rms), shape_quantiles=torch.quantile(torch.pdist(c / rms), torch.linspace(0, 1, 128)).numpy(), neighbor_bandwidth=bw, neighbor_mean=nm, neighbor_components=nc, neighbor_projection=nz, neighbor_gamma=ng)
    ts = tensors(s, 'cpu')
    with torch.no_grad():
        trials = [raw_terms(torch.from_numpy(r[:256]), ts, None if reference_coords is None else torch.as_tensor(reference_coords[:256], dtype=torch.float32))]
        for _ in range(3):
            ix = rng.choice(len(y), min(128, len(y) // 2), replace=False)
            trials.append(raw_terms(torch.from_numpy(y[ix]), ts, None if target_coords is None else torch.as_tensor(target_coords[ix], dtype=torch.float32)))
    s['scales'] = {k: 1.0 if k == 'direction' else max(0.0001, *[abs(float(t[k])) for t in trials]) for k in WEIGHTS}
    s['scale_calibration'] = [{k: float(v) for k, v in t.items()} for t in trials]
    return s

def balanced_objective(terms, stats, outputs):
    """Combine fixed priorities after bounded output-gradient balancing."""
    normalized = {k: v / stats['scales'][k] for k, v in terms.items()}
    norms = {}
    for k, v in normalized.items():
        grads = torch.autograd.grad(v, outputs, retain_graph=True, allow_unused=True)
        total = sum((g.detach().square().sum() * (stats.get('target_rms', 1.0) ** 2 if i == 1 else 1.0) for i, g in enumerate(grads) if g is not None))
        norms[k] = float(torch.sqrt(total)) if isinstance(total, torch.Tensor) else 0.0
    active = [v for v in norms.values() if v > 1e-12]
    median = float(np.median(active)) if active else 0.0
    factors = {k: float(np.clip(median / max(n, 1e-12), 0.1, 10)) if n > 1e-12 else 1.0 for k, n in norms.items()}
    value = sum((WEIGHTS[k] * factors[k] * v for k, v in normalized.items()))
    log = dict(raw={k: float(v.detach()) for k, v in terms.items()}, normalized={k: float(v.detach()) for k, v in normalized.items()}, output_gradient_norms=norms, balance_factors=factors, effective_weights={k: WEIGHTS[k] * factors[k] / stats['scales'][k] for k in terms})
    return (value, log)
