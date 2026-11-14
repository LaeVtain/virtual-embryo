"""Optional exact replay against the archived implementation and checkpoints.

Set VIRTUAL_EMBRYO_LEGACY_WORKSPACE to the original workspace to enable these
checks. They compare the actual all-gene networks, not a reduced architecture.
"""
import ast
import gc
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits

from virtual_embryo.t1.assets import export_assets
from virtual_embryo.t1.density import make_randomness
from virtual_embryo.t1.model import load_training_model, initialize_training_model
from virtual_embryo.t1.model_loader import PopulationPredictor
from virtual_embryo.t1.stream_predict import predict
from virtual_embryo.t1.train import elbo_step, teacher_step

WORKSPACE = os.environ.get('VIRTUAL_EMBRYO_LEGACY_WORKSPACE')
pytestmark = pytest.mark.skipif(not WORKSPACE, reason='Archived T1 workspace not selected')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def archive():
    workspace = Path(WORKSPACE)
    torch.set_num_threads(2)
    package = types.ModuleType('_t1_training_legacy')
    package.__path__ = [str(workspace/'T1_conditional_calibration/candidate/ve_model')]
    sys.modules[package.__name__] = package
    network = importlib.import_module('_t1_training_legacy.time_compositional_vae')
    density = importlib.import_module('_t1_training_legacy.sparse_compositional_vae')
    objectives = importlib.import_module('_t1_training_legacy.task_metric_losses')
    detection = importlib.import_module('_t1_training_legacy.prior_detection')
    clean = workspace/'T1_joint_moment_clean_20261112'
    replaced = {k: sys.modules.pop(k) for k in list(sys.modules)
                if k in ('model_loader', 'projection', 'moments', 've_model') or k.startswith('ve_model.')}
    sys.path.insert(0, str(clean))
    try:
        runtime = module_at('_t1_runtime_legacy', clean/'stream_predict.py')
    finally:
        sys.path.remove(str(clean))
        for key in list(sys.modules):
            if key in ('model_loader', 'projection', 'moments', 've_model') or key.startswith('ve_model.'):
                del sys.modules[key]
        sys.modules.update(replaced)
    asset_dir = workspace/'Cleanup_20261114/T1/assets'
    if not (asset_dir/'manifest.json').exists():
        export_assets(workspace, asset_dir)
    with threadpool_limits(limits=2):
        yield dict(workspace=workspace, network=network, density=density, objectives=objectives,
                   detection=detection, runtime=runtime, assets=asset_dir)


def test_five_real_checkpoints_and_all_prediction_routes(archive, tmp_path):
    old = archive['runtime'].load_model()
    new = PopulationPredictor(archive['assets'])
    data = sparse.load_npz(archive['workspace']/'T1_prior_detection/data/train_9.5.npz')[:4]
    x = torch.from_numpy(data.toarray())
    noise = make_randomness(4, new.n_genes, 32, 101)
    for name in ('original', 'fullELBO30', 'teacher11', 'teacher22', 'teacher33'):
        for start, end in ((8.5, 9.5), (9.5, 10.5), (10.5, 12.5)):
            expected = old.model(name).sample_prior(x, noise, start, end, output_dtype=torch.float64)
            actual = new.model(name).sample_prior(x, noise, start, end, output_dtype=torch.float64)
            assert torch.equal(actual, expected), (name, start, end)
    for index, (start, end) in enumerate(((8.5, 9.5), (9.5, 10.5), (8.5, 10.5))):
        expected, _ = archive['runtime'].predict(old, data, start, end, 101, tmp_path/f'old{index}')
        actual, _ = predict(new, data, start, end, 101, tmp_path/f'new{index}')
        assert np.array_equal(actual, expected), (start, end)
    del old, new
    gc.collect()


def reference_teacher_update(archive, net, optimizer, x, y, epsilon, noise_seed, moments, balance):
    """Execute the original training body's statements through opt.step()."""
    text = (archive['workspace']/'T1_target95_105_20261109/run.py').read_text(encoding='utf-8')
    tree = ast.parse(text)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in train.body if isinstance(n, ast.For))
    begin = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'elbo' for t in n.targets))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
               and ast.unparse(n.test).startswith("arm == 'dv50long'"))
    body = loop.body[begin:end]
    # These two archived metric calculations are disconnected from the loss.
    # Stub them so the replay needs no historical PCA/statistics artifacts.
    old = types.SimpleNamespace(make_randomness=archive['density'].make_randomness,
                                losses=types.SimpleNamespace(
                                    prior_sample_fixed_support=archive['objectives'].prior_sample_fixed_support,
                                    raw_terms=lambda *args: {}))
    namespace = dict(torch=torch, net=net, opt=optimizer, x=x, y=y, a=9.5, b=10.5, eps=epsilon,
                     params=list(net.parameters()), arm='dv50long', i=1, ns=noise_seed, old=old,
                     stats=[None, dict(pca_mean=torch.zeros(net.n_genes),
                                      pca_components=torch.zeros(1, net.n_genes))],
                     moments=[None, moments], scales=balance['scales'], base_initial=balance['base_initial'],
                     prior_detection_rates=archive['detection'].prior_detection_rates,
                     norm=lambda gs: sum(float(g.square().sum()) for g in gs if g is not None)**.5,
                     records=[], step=1)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<archived teacher update>', 'exec'), namespace)
    balance['base_initial'] = namespace['base_initial']
    return namespace['record']


def test_real_elbo_loss_and_teacher_gradient_updates(archive):
    workspace = archive['workspace']
    checkpoint = torch.load(workspace/'T1_joint_moment_clean_20261112/weights/fullELBO30.pt',
                            map_location='cpu', weights_only=False)
    old = archive['network'].TimeConditionedCompositionalVAE(record=checkpoint['record'], **checkpoint['config'])
    old.load_state_dict(checkpoint['model'], strict=True)
    new = load_training_model(archive['assets']/'training/fullELBO30.pt')
    del checkpoint
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    x = torch.from_numpy(sparse.load_npz(workspace/'T1_prior_detection/data/train_9.5.npz')[:64].toarray())
    y = torch.from_numpy(sparse.load_npz(workspace/'T1_prior_detection/data/train_10.5.npz')[:64].toarray())
    epsilon = torch.randn(64, 32, generator=torch.Generator().manual_seed(2710001))
    old_optimizer = torch.optim.Adam(old.parameters(), lr=1e-5)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=1e-5)
    # First ELBO update uses a few real rows; all loss terms and gradients agree.
    expected = old.loss(x[:4], y[:4], 9.5, 10.5, epsilon[:4])
    actual = new.loss(x[:4], y[:4], 9.5, 10.5, epsilon[:4])
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    old_optimizer.zero_grad()
    expected['negative_elbo_per_gene'].mean().backward()
    torch.nn.utils.clip_grad_norm_(old.parameters(), 10, error_if_nonfinite=True)
    old_optimizer.step()
    elbo_step(new, new_optimizer, x[:4], y[:4], 9.5, 10.5, epsilon[:4])
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    del expected, actual
    moments = dict(rate=(y > 0).float().mean(0), variance=y.var(0, unbiased=True))
    old_balance, new_balance = (dict(scales={}, base_initial=None) for _ in range(2))
    # Two steps protect both first-use scale fitting and persistent scale reuse.
    for noise_seed in (2210001, 2210004):
        expected = reference_teacher_update(archive, old, old_optimizer, x, y, epsilon,
                                            noise_seed, moments, old_balance)
        actual = teacher_step(new, new_optimizer, x, y, 1, epsilon, noise_seed, moments, new_balance)
        for key in ('elbo', 'auxiliary', 'moment_factor', 'grad_norm', 'losses'):
            assert actual[key] == expected[key], key
        for key, value in old.state_dict().items():
            assert torch.equal(value, new.state_dict()[key]), key
    del old, new, old_optimizer, new_optimizer
    gc.collect()


def test_control_initialization_and_first_update(archive):
    target = np.log1p(10000*np.random.default_rng(5).dirichlet(np.ones(11), size=8)).astype('float32')
    record = archive['density'].fit_initialization(target)
    torch.manual_seed(1)
    old = archive['network'].TimeConditionedCompositionalVAE(n_genes=11, record=record, hidden=8, latent_dim=4)
    torch.manual_seed(1)
    new = initialize_training_model(record, hidden=8, latent_dim=4)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    x, y = torch.from_numpy(target[:4]), torch.from_numpy(target[4:])
    epsilon = torch.randn(4, 4, generator=torch.Generator().manual_seed(8803))
    old_opt, new_opt = torch.optim.Adam(old.parameters(), lr=.001), torch.optim.Adam(new.parameters(), lr=.001)
    loss = old.loss(x, y, 8.5, 9.5, epsilon)['negative_elbo_per_gene'].mean()
    old_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(old.parameters(), 10, error_if_nonfinite=True)
    old_opt.step()
    actual = elbo_step(new, new_opt, x, y, 8.5, 9.5, epsilon)
    assert actual['elbo'] == float(loss.detach())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
