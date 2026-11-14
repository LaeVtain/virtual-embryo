"""Final T1 teacher moment objectives."""
import torch
from torch.nn import functional as F
from .density import sample_observation, SamplingSupportError

def prior_sample_fixed_support(net, source, start, end, noise, *, allow_cast_underflow=False):
    """Exact hard sampler with pathwise intensity/z gradients, no gate estimator.

    Boolean sampling decisions already detach gate logits in sample_observation.
    Gradients may still change shared features; this does NOT freeze future masks.
    """
    h = net._condition(source, start, end)
    prior = net._gaussian(net.prior(h))
    z = prior['mean'].double() + (.5*prior['logvar'].double()).exp()*noise['latent_normal'].to(h.device)
    params = net._decode(h, z)
    params['gate_logits'] = params['gate_logits'].detach()
    random={k:v.to(h.device) for k,v in noise.items() if k!='latent_normal'}
    try:
        hard,audit=sample_observation(params,random)
    except SamplingSupportError as exc:
        d=getattr(exc,'diagnostics',{})
        if not (allow_cast_underflow and d.get('cast_positive_underflow_entries',0)>0
                and d.get('positive_underflow_entries')==0 and d.get('nonfinite_entries')==0):
            raise
        # Preserve the sampled support in float64. Loss arithmetic remains
        # float32; conversion uses ordinary IEEE rounding, no chosen threshold.
        # The strict sampler and public prediction default remain unchanged.
        hard,audit=sample_observation(params,random,output_dtype=torch.float64)
        lost=(hard>0)&(hard.float()==0)
        audit.update(loss_float32_underflow_entries=int(lost.sum()),
                     loss_float32_underflow_mass=float(torch.expm1(hard[lost]).sum()),
                     support_retained_in_float64=True)
    # The official sampler is decorated no_grad. Rebuild only its continuous
    # conditional intensity law, using the EXACT sampled, detached support.
    logits=params['location'].double()+params['scale'].double()*random['positive_normal'].double()
    active=torch.where(hard>0,logits,torch.full_like(logits,-torch.inf))
    continuous=torch.log1p(10000*(active-torch.logsumexp(active,-1,keepdim=True)).exp()).float()
    return hard.float()+(continuous-continuous.detach()),audit

def conditional_detection_probabilities(gate_logits):
    """Return P(M_g=1 | M is nonempty) in float64 for each row.

    Z = sum_g pi_g prod_{j<g}(1-pi_j) partitions the nonempty event by its
    first success. Its logsumexp evaluation avoids both 1-product cancellation
    and sigmoid underflow at very negative logits. No clipping is performed.
    """
    if (not isinstance(gate_logits, torch.Tensor)
            or not gate_logits.is_floating_point() or gate_logits.ndim != 2
            or min(gate_logits.shape) < 1 or not torch.isfinite(gate_logits).all()):
        raise ValueError("Finite nonempty floating rows-by-genes logits required")
    logits = gate_logits.double()
    log_pi = F.logsigmoid(logits)
    log_no = F.logsigmoid(-logits)
    prefix = torch.cat((torch.zeros_like(log_no[:, :1]),
                        log_no.cumsum(dim=-1)[:, :-1]), dim=-1)
    log_z = torch.logsumexp(log_pi + prefix, dim=-1, keepdim=True)
    return torch.exp(log_pi - log_z)

def prior_detection_rates(net, source, t0, t1, epsilon):
    """Average analytic detection probabilities over source rows and prior z.

    epsilon has shape [K, B, latent_dim], is explicit standard-Normal base
    noise, and must be on the source device. Each K draw covers all B rows.
    The returned vector covers ALL model genes. This routine never uses a
    target expression or the training posterior.

    Gradients reach the source/time encoders, prior and decoder.
    """
    if (not isinstance(epsilon, torch.Tensor) or not epsilon.is_floating_point()
            or epsilon.ndim != 3 or epsilon.shape[0] < 1
            or epsilon.shape[1:] != (len(source), net.latent_dim)
            or epsilon.device != source.device or not torch.isfinite(epsilon).all()
            or epsilon.requires_grad):
        raise ValueError("Explicit finite independent base noise [K,B,Z] required")
    h = net._condition(source, t0, t1)
    prior = net._gaussian(net.prior(h))
    mean = prior['mean'].double()
    std = torch.exp(.5 * prior['logvar'].double())
    rates = []
    for noise in epsilon:
        z = mean + std * noise.double()
        parameters = net._decode(h, z)
        rates.append(conditional_detection_probabilities(
            parameters['gate_logits']).mean(dim=0))
    return torch.stack(rates, dim=0).mean(dim=0)
