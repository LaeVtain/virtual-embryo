import torch
import numpy as np
from virtual_embryo.t1.density import make_randomness
from virtual_embryo.t1.model import TrainingModel, TransitionModel, inference_state
from virtual_embryo.t1.stream_predict import noise_blocks
from virtual_embryo.t1.train import elbo_step, teacher_step


def test_independent_streams_keep_row_order():
    expected = make_randomness(65, 11, 4, 101)
    blocks = list(noise_blocks(65, 11, 4, 101, batch=32))
    for name, value in expected.items():
        assert torch.equal(value, torch.cat([noise[name] for _, noise in blocks]))


def test_training_and_deployment_share_the_prior():
    torch.manual_seed(42)
    train = TrainingModel(n_genes=11, hidden=8, latent_dim=4)
    deploy = TransitionModel(n_genes=11, hidden=8, latent_dim=4)
    deploy.load_state_dict(inference_state(train.state_dict()), strict=True)
    x = torch.log1p(10000*torch.softmax(torch.randn(5, 11), -1))
    noise = make_randomness(5, 11, 4, 101)
    assert torch.equal(train.sample_prior(x, noise, 9.5, 10.5, output_dtype=torch.float64),
                       deploy.sample_prior(x, noise, 9.5, 10.5, output_dtype=torch.float64))
    assert not any(k.startswith(('posterior.', 'target_encoder.')) for k in deploy.state_dict())


def test_both_training_objectives_update_the_posterior_and_prior():
    torch.manual_seed(42)
    net = TrainingModel(n_genes=11, hidden=8, latent_dim=4)
    x = torch.log1p(10000*torch.softmax(torch.randn(64, 11), -1))
    y = torch.log1p(10000*torch.softmax(torch.randn(64, 11), -1))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    epsilon = torch.randn(64, 4)
    first = net.posterior[-1].weight.detach().clone()
    record = elbo_step(net, optimizer, x, y, 8.5, 9.5, epsilon)
    assert np.isfinite(record['elbo'])
    assert not torch.equal(first, net.posterior[-1].weight)
    balance = dict(scales={}, base_initial=None)
    record = teacher_step(net, optimizer, x, y, 1, epsilon, 2210001,
                          dict(variance=y.var(0, unbiased=True), rate=(y > 0).float().mean(0)), balance)
    assert set(record['losses']) == {'variance', 'detection'}
    assert 0 < record['moment_factor'] <= 1
    assert torch.isfinite(net.prior[-1].weight.grad).all()
