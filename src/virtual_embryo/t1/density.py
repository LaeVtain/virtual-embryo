"""Sparse composition likelihood and independent sampling streams."""
import math
import numpy as np
import torch
from torch.nn import functional as F

TOTAL=10000.
SIGMA_MIN,SIGMA_MAX=.05,3.
LOGVAR_MIN,LOGVAR_MAX=-6.,4.

class SamplingSupportError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics

def _matrix(x, name, genes=None):
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.ndim != 2
            or min(x.shape) < 1 or not x.is_floating_point()
            or (genes is not None and x.shape[1] != genes)):
        raise ValueError(name+" must be a complete floating row-by-gene tensor")
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError(name+" must be finite and nonnegative")

def observed_composition(target):
    _matrix(target, "target")
    x = target.double()
    a = torch.expm1(x)
    totals = a.sum(-1, keepdim=True)
    if not torch.isfinite(a).all() or not torch.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Target composition must have finite positive row totals")
    mask = x > 0
    p = a/totals
    if (p[mask] <= 0).any():
        raise ValueError("Positive target composition underflow")
    return p, mask

def nonempty_log_normalizer(gate_logits):
    """log(1-prod(1-pi)), stable also when all logits are extremely negative."""
    logits = gate_logits.double()
    if logits.ndim != 2 or min(logits.shape) < 1 or not torch.isfinite(logits).all():
        raise ValueError("Finite matrix of gate logits required")
    log_no = F.logsigmoid(-logits)
    log_empty = log_no.sum(-1)
    # Safe operands prevent autograd NaNs in the unused exact-zero branch.
    safe = torch.where(log_empty < 0, log_empty, torch.full_like(log_empty, -1.))
    logz = torch.log(-torch.expm1(safe))
    prefix = torch.cat((torch.zeros_like(log_no[:, :1]), log_no.cumsum(-1)[:, :-1]), dim=-1)
    fallback = torch.logsumexp(F.logsigmoid(logits)+prefix, dim=-1)
    return torch.where(log_empty < 0, logz, fallback)

def nonempty_mask_nll(gate_logits, mask):
    if mask.shape != gate_logits.shape or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("Observed support must be aligned, boolean and nonempty")
    logits = gate_logits.double()
    return F.binary_cross_entropy_with_logits(logits, mask.double(), reduction="none").sum(-1) + nonempty_log_normalizer(logits)

def logistic_normal_nll(proportions, mask, location, scale):
    """Exact density on each active (k-1)-simplex; no reference gene chosen.

    The return value is a differential NLL in composition coordinates and can be
    negative. It is not a G-dimensional density with respect to raw log X.
    """
    if not (proportions.shape == mask.shape == location.shape == scale.shape) or mask.dtype != torch.bool:
        raise ValueError("Composition, support and all logits must have aligned complete axes")
    p, mu, sigma = proportions.double(), location.double(), scale.double()
    if (not torch.isfinite(p).all() or not torch.isfinite(mu).all() or not torch.isfinite(sigma).all()
            or (sigma <= 0).any() or (p[mask] <= 0).any() or (p[~mask] != 0).any()
            or not mask.any(-1).all() or not torch.allclose(p.sum(-1), torch.ones_like(p[:, 0]), rtol=0, atol=1e-10)):
        raise ValueError("Illegal sparse continuous composition or Normal parameters")
    logp = torch.log(torch.where(mask, p, torch.ones_like(p)))
    weights = torch.where(mask, sigma.reciprocal().square(), torch.zeros_like(sigma))
    residual = logp-mu
    center = (weights*residual).sum(-1, keepdim=True)/weights.sum(-1, keepdim=True)
    quad = (weights*(residual-center).square()).sum(-1)
    logdet = torch.where(mask, 2*sigma.log(), torch.zeros_like(sigma)).sum(-1)+weights.sum(-1).log()
    k = mask.sum(-1)
    nll = .5*((k.to(p.dtype)-1)*math.log(2*math.pi)+logdet+quad)+torch.where(mask, logp, torch.zeros_like(logp)).sum(-1)
    return torch.where(k == 1, torch.zeros_like(nll), nll)

def make_randomness(rows, genes, latent_dim=32, seed=0):
    """CPU float64 independent base streams; slicing rows preserves every draw."""
    if any(type(v) is not int or v < 1 for v in (rows, genes, latent_dim)) or type(seed) is not int or seed < 0:
        raise ValueError("Positive dimensions and nonnegative integer seed required")
    shapes = dict(latent_normal=(rows,latent_dim), mask_first_uniform=(rows,),
                  mask_uniform=(rows,genes), positive_normal=(rows,genes))
    result = {}
    for stream, (name, shape) in enumerate(shapes.items()):
        rng = np.random.default_rng(np.random.SeedSequence([seed, stream, 9711]))
        array = rng.standard_normal(shape) if name.endswith("normal") else rng.random(shape)
        result[name] = torch.from_numpy(array)
    return result

def _noise(randomness, rows, genes, latent_dim, device):
    shapes = dict(latent_normal=(rows,latent_dim), mask_first_uniform=(rows,),
                  mask_uniform=(rows,genes), positive_normal=(rows,genes))
    if set(randomness) != set(shapes):
        raise ValueError("Exactly four explicit independent base-noise arrays are required")
    out = {}
    for name, shape in shapes.items():
        x = torch.as_tensor(randomness[name], device=device, dtype=torch.float64)
        if tuple(x.shape) != shape or not torch.isfinite(x).all():
            raise ValueError("Invalid base randomness: "+name)
        if name.endswith("uniform") and ((x < 0).any() or (x >= 1).any()):
            raise ValueError("Uniform base noise must lie in [0,1)")
        out[name] = x
    return out

@torch.no_grad()
def sample_observation(parameters, randomness, *, total=TOTAL, output_dtype=torch.float32, audit=True):
    """Sample the exact nonempty-mask/active-logistic-normal observation law.

    Required noise keys here: mask_first_uniform, mask_uniform, positive_normal.
    A first-success draw followed by independent tail Bernoulli draws samples a
    product Bernoulli law conditional on nonempty support, without rejection or
    forcibly modifying an unconditional sample.
    """
    gate, mu, sigma = [parameters[k].double() for k in ("gate_logits", "location", "scale")]
    if audit:
        if not (gate.shape == mu.shape == sigma.shape) or gate.ndim != 2 or min(gate.shape) < 1:
            raise ValueError("Aligned nonempty observation parameters required")
        if not all(torch.isfinite(v).all() for v in (gate,mu,sigma)) or (sigma <= 0).any():
            raise ValueError("Finite Normal/gate parameters and positive scales required")
        if total != TOTAL or output_dtype not in (torch.float32, torch.float64):
            raise ValueError("Only the frozen total and float32/float64 output are supported")
    n, g = gate.shape
    checked = (_noise(dict(randomness, latent_normal=torch.zeros((n,1))), n, g, 1, gate.device)
               if audit else randomness)
    log_no = F.logsigmoid(-gate)
    prefix = torch.cat((torch.zeros_like(log_no[:, :1]), log_no.cumsum(-1)[:, :-1]), -1)
    first_logweight = F.logsigmoid(gate)+prefix
    logcdf = torch.logcumsumexp(first_logweight, -1)-torch.logsumexp(first_logweight, -1, keepdim=True)
    logu = checked["mask_first_uniform"].log()
    first = (logcdf < logu[:, None]).sum(-1)
    if audit and (first >= g).any():
        raise SamplingSupportError("Nonempty categorical draw escaped support", {"invalid_first_indices":int((first>=g).sum())})
    index = torch.arange(g, device=gate.device)[None, :]
    tail = checked["mask_uniform"].log() < F.logsigmoid(gate)
    mask = (index == first[:,None]) | ((index > first[:,None]) & tail)
    logits = mu+sigma*checked["positive_normal"]
    active = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    logp = active-torch.logsumexp(active, -1, keepdim=True)
    p = logp.exp()
    x64 = torch.log1p(total*p)
    result = x64.to(output_dtype)
    if not audit:
        return result, {}
    underflow = mask & ((p <= 0) | (x64 <= 0))
    cast_underflow = mask & (result <= 0)
    diagnostics = dict(rows=n, genes=g, nonempty_sampling="first_success_then_independent_tail",
        zero_rows=int((~mask.any(-1)).sum()), single_positive_rows=int((mask.sum(-1)==1).sum()),
        positive_underflow_entries=int(underflow.sum()), cast_positive_underflow_entries=int(cast_underflow.sum()),
        nonfinite_entries=int((~torch.isfinite(result)).sum()), output_dtype=str(result.dtype),
        total=total, maximum_relative_total_error=float((torch.expm1(result.double()).sum(-1)/total-1).abs().max()),
        normalizing_total_is_constructed=True, rejection_or_support_repair=False)
    if underflow.any() or cast_underflow.any() or not torch.isfinite(result).all() or diagnostics["zero_rows"]:
        raise SamplingSupportError("Illegal/underflowed sampled positive support; no repair", diagnostics)
    if (result[~mask] != 0).any() or diagnostics["maximum_relative_total_error"] > 2e-6:
        raise SamplingSupportError("Sampled observation violates the frozen composition convention", diagnostics)
    return result, diagnostics
