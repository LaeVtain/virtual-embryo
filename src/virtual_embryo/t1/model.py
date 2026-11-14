"""Shared time-conditioned network; training adds the posterior modules."""
import torch
from torch import nn

from .density import (
    LOGVAR_MIN, LOGVAR_MAX, SIGMA_MIN, SIGMA_MAX, observed_composition,
    nonempty_mask_nll, logistic_normal_nll, sample_observation,
)


class TransitionModel(nn.Module):
    def __init__(self, n_genes, hidden=256, latent_dim=32, total=10000., *, training=False):
        super().__init__()
        self.n_genes, self.hidden, self.latent_dim = n_genes, hidden, latent_dim
        self.total = total
        self.model_config = dict(n_genes=n_genes, hidden=hidden, latent_dim=latent_dim, total=total)

        def encoder():
            return nn.Sequential(nn.Linear(n_genes, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())

        # Preserve the original training parameter order: the auxiliary gradient
        # budget sums squared norms in this order.
        self.source_encoder = encoder()
        if training:
            self.target_encoder = encoder()
        self.prior = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2*latent_dim))
        if training:
            self.posterior = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2*latent_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden+latent_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3*n_genes))
        self.time_encoder = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def _source(self, source):
        return self.source_encoder(source.to(self.decoder[-1].weight.dtype))

    def _condition(self, source, source_time, target_time):
        h = self._source(source)

        def vector(value):
            v = torch.as_tensor(value, device=h.device, dtype=h.dtype)
            return v.expand(len(h)) if v.ndim == 0 else v

        start, end = vector(source_time), vector(target_time)
        return h+self.time_encoder(torch.stack((start-8.5, end-start), dim=-1))

    def _gaussian(self, raw):
        mu, raw_logvar = raw.chunk(2, -1)
        return dict(mean=mu, logvar=LOGVAR_MIN+(LOGVAR_MAX-LOGVAR_MIN)*raw_logvar.sigmoid())

    def _decode(self, h, z):
        gate, mu, raw_scale = self.decoder(torch.cat((h, z.to(h.dtype)), -1)).chunk(3, -1)
        return dict(gate_logits=gate, location=mu, scale=SIGMA_MIN+(SIGMA_MAX-SIGMA_MIN)*raw_scale.sigmoid())

    @torch.no_grad()
    def sample_prior(self, source, randomness, source_time, target_time, *, output_dtype=torch.float32):
        h = self._condition(source, source_time, target_time)
        gaussian = self._gaussian(self.prior(h))
        z = gaussian['mean'].double()+torch.exp(.5*gaussian['logvar'].double())*randomness['latent_normal']
        result, _ = sample_observation(
            self._decode(h, z), {k: v for k, v in randomness.items() if k != 'latent_normal'},
            total=self.total, output_dtype=output_dtype, audit=False,
        )
        return result


class TrainingModel(TransitionModel):
    def __init__(self, **config):
        super().__init__(**config, training=True)

    def loss(self, source, target, source_time, target_time, epsilon):
        h = self._condition(source, source_time, target_time)
        prior = self._gaussian(self.prior(h))
        post = self._gaussian(self.posterior(torch.cat((h, self.target_encoder(target.to(h.dtype))), -1)))
        z = post['mean']+torch.exp(.5*post['logvar'])*epsilon.to(h.dtype)
        pm, pl, qm, ql = [v.double() for v in (prior['mean'], prior['logvar'], post['mean'], post['logvar'])]
        kl_dimension = .5*(pl-ql+(ql-pl).exp()+(qm-pm).square()*(-pl).exp()-1)
        parameters = self._decode(h, z)
        proportions, mask = observed_composition(target)
        mask_nll = nonempty_mask_nll(parameters['gate_logits'], mask)
        composition_nll = logistic_normal_nll(proportions, mask, parameters['location'], parameters['scale'])
        kl = kl_dimension.sum(-1)
        observation_nll = mask_nll+composition_nll
        total = observation_nll+kl
        if not torch.isfinite(total).all():
            raise ValueError('Nonfinite ELBO')
        return dict(mask_nll=mask_nll, composition_nll=composition_nll, kl=kl,
                    kl_per_dimension=kl_dimension, observation_nll=observation_nll,
                    total_nll=total, negative_elbo=total, negative_elbo_per_gene=total/self.n_genes)


def checkpoint_config(checkpoint):
    """Read either an original checkpoint or the portable checkpoint format."""
    config = checkpoint['config']
    if config.get('arm', 'shared_latent') != 'shared_latent' or not config.get('use_time', True):
        raise ValueError('The final T1 recipe requires shared_latent with time conditioning')
    return dict(n_genes=config['n_genes'], hidden=config.get('hidden', 256),
                latent_dim=config.get('latent_dim', 32),
                total=config.get('total', checkpoint.get('record', {}).get('total', 10000.)))


def inference_state(state):
    return {k: v for k, v in state.items() if not k.startswith(('target_encoder.', 'posterior.'))}


def load_training_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = TrainingModel(**checkpoint_config(checkpoint))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


def initialize_training_model(record, *, hidden=256, latent_dim=32):
    """Original control initialization, including its fitted observation head."""
    net = TrainingModel(n_genes=record['n_genes'], hidden=hidden, latent_dim=latent_dim, total=record['total'])
    for network in (net.prior, net.posterior):
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        with torch.no_grad():
            network[-1].bias[latent_dim:].fill_(torch.logit(torch.tensor(.6)).item())
    nn.init.zeros_(net.decoder[-1].weight)
    probability = torch.tensor(record['detection_probability'])
    sigma = torch.tensor(record['initial_sigma'])
    bias = torch.cat((torch.logit(probability), torch.tensor(record['location']),
                      torch.logit((sigma-SIGMA_MIN)/(SIGMA_MAX-SIGMA_MIN))))
    with torch.no_grad():
        net.decoder[-1].bias.copy_(bias)
    return net
