"""Final fullELBO30 and dv50long continuation-training recipes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from ..paths import task_assets
from .density import make_randomness
from .model import load_training_model, initialize_training_model, inference_state
from .objectives import prior_sample_fixed_support, prior_detection_rates

TIMES = (8.5, 9.5, 10.5)
PAIRS = ((8.5, 9.5), (9.5, 10.5), (8.5, 10.5))
PAIR_INDICES = ((0, 1), (1, 2), (0, 2))
SEEDS = (11, 22, 33)


def norm(gradients):
    return sum(float(g.square().sum()) for g in gradients if g is not None)**.5


def elbo_step(net, optimizer, x, y, start, end, epsilon):
    parameters = list(net.parameters())
    loss = net.loss(x, y, start, end, epsilon)['negative_elbo_per_gene'].mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
    optimizer.step()
    return dict(elbo=float(loss.detach()), grad_norm=float(grad_norm))


def teacher_step(net, optimizer, x, y, pair, epsilon, noise_seed, moments, balance):
    """One original dv50long update, including its persistent gradient scales."""
    start, end = PAIRS[pair]
    parameters = list(net.parameters())
    elbo = net.loss(x, y, start, end, epsilon)['negative_elbo_per_gene'].mean()
    optimizer.zero_grad(set_to_none=True)
    record = dict(pair=[start, end], elbo=float(elbo.detach()))
    if pair != 1:
        elbo.backward()
    else:
        pred, _ = prior_sample_fixed_support(
            net, x, start, end, make_randomness(len(x), net.n_genes, net.latent_dim, noise_seed),
            allow_cast_underflow=True,
        )
        auxiliary = elbo.new_zeros(())
        base_gradients = torch.autograd.grad(elbo, parameters, allow_unused=True)
        base_norm = norm(base_gradients)
        if balance['base_initial'] is None:
            balance['base_initial'] = base_norm
        epsilon_rate = torch.randn(2, min(32, len(x)), net.latent_dim,
                                   generator=torch.Generator().manual_seed(noise_seed+800000))
        rate = prior_detection_rates(net, x[:32], start, end, epsilon_rate)
        losses = dict(variance=(pred.var(0, unbiased=True)-moments['variance']).abs().mean(),
                      detection=(rate-moments['rate']).abs().mean())
        scales = balance['scales']
        for key, value in losses.items():
            if (pair, key) not in scales:
                gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                scales[pair, key] = .5*.5*base_norm/max(norm(gradients), 1e-12)
        moment = sum(value*scales[pair, key] for key, value in losses.items())
        moment_gradients = torch.autograd.grad(moment, parameters, retain_graph=True, allow_unused=True)
        factor = min(1., .5*max(base_norm, .1*balance['base_initial'])/max(norm(moment_gradients), 1e-12))
        auxiliary = auxiliary+factor*moment
        auxiliary_gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        for parameter, base, extra in zip(parameters, base_gradients, auxiliary_gradients):
            parameter.grad = (base if base is not None else 0)+(extra if extra is not None else 0)
        record.update(moment_factor=factor, losses={k: float(v.detach()) for k, v in losses.items()},
                      auxiliary=float(auxiliary.detach()))
    record['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True))
    optimizer.step()
    return record


def full_elbo_batches(data, prepared, seed=22, epochs=30):
    probe = joblib.load(prepared/'probe.joblib')
    codes = [probe['svd'].transform(data[t]).astype('float32') for t in TIMES]
    plans = {}
    for i, j in PAIR_INDICES:
        temperature = json.loads((prepared/f'coupling_{i}_{j}.json').read_text())['kernel']['temperature']
        probability = torch.softmax(torch.from_numpy(-cdist(codes[j], codes[i], 'sqeuclidean')/temperature), dim=1).numpy()
        cdf = probability.cumsum(1)
        cdf[:, -1] = 1
        plans[i, j] = cdf
    for epoch in range(epochs):
        for i, j in PAIR_INDICES:
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 5101])).permutation(1024)
            uniform = np.random.default_rng(np.random.SeedSequence([seed, epoch, 5102])).random(1024)
            ids = np.array([np.searchsorted(plans[i, j][row], value, side='right') for row, value in zip(order, uniform)])
            for offset in range(0, 1024, 128):
                yield epoch, i, j, ids[offset:offset+128], order[offset:offset+128]


def teacher_batches(prepared, seed):
    epoch = SEEDS.index(seed)
    schedules = [dict(np.load(prepared/f'pairs_{i}_{j}.npz')) for i, j in PAIR_INDICES]
    for _ in range(3):
        for batch in range(16):
            for index, schedule in enumerate(schedules):
                selected = slice(batch*64, (batch+1)*64)
                yield index, schedule['source'][epoch, selected], schedule['target'][epoch, selected]


def save_training(path, net, optimizer, records, **state):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=net.state_dict(), config=net.model_config,
                    optimizer=optimizer.state_dict(), **state), path/'model.pt')
    torch.save(dict(model=inference_state(net.state_dict())), path/'inference.pt')
    (path/'training.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def train_control(data_dir, prepared, output, *, initialization=None, epochs=12, max_steps=None, device='cpu'):
    """Train the original control from fitted initialization and full schedules."""
    began = time.monotonic()
    data = {t: sparse.load_npz(data_dir/f'train_{t}.npz') for t in TIMES}
    record = json.loads((initialization or prepared/'initialization.json').read_text(encoding='utf-8'))
    schedules = {(i, j): dict(np.load(prepared/f'pairs_{i}_{j}.npz')) for i, j in PAIR_INDICES}
    if any(len(schedule['source']) < epochs for schedule in schedules.values()):
        raise ValueError('Control training requires one source/target schedule per epoch')
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    net = initialize_training_model(record).to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(8803)
    records = []
    for epoch in range(epochs):
        for i, j in PAIR_INDICES:
            source_rows, target_rows = schedules[i, j]['source'][epoch], schedules[i, j]['target'][epoch]
            epsilon = torch.randn(len(target_rows), net.latent_dim, generator=rng)
            for offset in range(0, len(target_rows), 128):
                if max_steps is not None and len(records) >= max_steps:
                    break
                si, ti = source_rows[offset:offset+128], target_rows[offset:offset+128]
                x = torch.from_numpy(data[TIMES[i]][si].toarray()).to(device)
                y = torch.from_numpy(data[TIMES[j]][ti].toarray()).to(device)
                item = elbo_step(net, optimizer, x, y, TIMES[i], TIMES[j], epsilon[offset:offset+len(ti)].to(device))
                item.update(step=len(records)+1, epoch=epoch+1, pair=[TIMES[i], TIMES[j]])
                records.append(item)
            if max_steps is not None and len(records) >= max_steps:
                break
        if max_steps is not None and len(records) >= max_steps:
            break
        print(f'control epoch {epoch+1}/{epochs}', flush=True)
    save_training(output, net, optimizer, records, seed=1, updates=len(records), rng_state=rng.get_state())
    return dict(recipe='control', seed=1, updates=len(records), seconds=time.monotonic()-began)


def train_full_elbo(data_dir, prepared, checkpoint, output, *, seed=22, epochs=30, max_steps=None):
    began = time.monotonic()
    net = load_training_model(checkpoint).train().requires_grad_(True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    data = {t: sparse.load_npz(data_dir/f'train_{t}.npz').toarray().astype('float32') for t in TIMES}
    if any(len(x) != 1024 for x in data.values()):
        raise ValueError('The frozen fullELBO30 recipe uses 1024 training cells per stage')
    rng = torch.Generator().manual_seed(8803+seed)
    records = []
    for step, (epoch, i, j, source_rows, target_rows) in enumerate(full_elbo_batches(data, prepared, seed, epochs)):
        if max_steps is not None and step >= max_steps:
            break
        x = torch.from_numpy(data[TIMES[i]][source_rows])
        y = torch.from_numpy(data[TIMES[j]][target_rows])
        record = elbo_step(net, optimizer, x, y, TIMES[i], TIMES[j], torch.randn(len(x), net.latent_dim, generator=rng))
        record.update(step=step+1, epoch=epoch+1, pair=[TIMES[i], TIMES[j]])
        records.append(record)
        if (step+1) % 24 == 0:
            print(f'fullELBO epoch {epoch+1}/{epochs}', flush=True)
    save_training(output, net, optimizer, records, rng_state=rng.get_state(), seed=seed, updates=len(records))
    return dict(recipe='fullELBO30', seed=seed, updates=len(records), seconds=time.monotonic()-began)


def train_teacher(data_dir, prepared, checkpoint, output, *, seed, max_steps=None):
    began = time.monotonic()
    net = load_training_model(checkpoint).train().requires_grad_(True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    data = {t: sparse.load_npz(data_dir/f'train_{t}.npz') for t in TIMES}
    target = data[10.5].toarray().astype('float32')
    moments = dict(rate=torch.from_numpy((target > 0).mean(0).astype('float32')),
                   variance=torch.from_numpy(target.var(0, ddof=1)))
    balance = dict(scales={}, base_initial=None)
    trace, records = hashlib.sha256(), []
    for step, (pair, source_rows, target_rows) in enumerate(teacher_batches(prepared, seed)):
        if max_steps is not None and step >= max_steps:
            break
        start, end = PAIRS[pair]
        noise_seed = 2100000+seed*10000+step
        epsilon = torch.randn(64, net.latent_dim, generator=torch.Generator().manual_seed(noise_seed+500000))
        for value in (source_rows, target_rows, epsilon.numpy(), np.array([pair, noise_seed])):
            trace.update(np.ascontiguousarray(value).tobytes())
        x, y = torch.from_numpy(data[start][source_rows].toarray()), torch.from_numpy(data[end][target_rows].toarray())
        record = teacher_step(net, optimizer, x, y, pair, epsilon, noise_seed, moments, balance)
        record['step'] = step+1
        records.append(record)
    save_training(output, net, optimizer, records, seed=seed, updates=len(records), balance=balance)
    return dict(recipe='dv50long', seed=seed, updates=len(records), trace_sha256=trace.hexdigest(),
                seconds=time.monotonic()-began)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=('control', 'full-elbo', 'teacher', 'all'))
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path, default=task_assets('t1'))
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int, help='Default: control 12; full-elbo 30')
    parser.add_argument('--initialization', type=Path, help='Control observation-head initialization JSON')
    parser.add_argument('--device', default='cpu', help='Device for original control training')
    parser.add_argument('--max-steps', type=int, help='Limit updates for a short local verification run')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared_dir or args.model_dir/'prepared'
    results = []
    if args.recipe == 'control':
        if args.prepared_dir is None:
            parser.error('control requires --prepared-dir with the original full training schedules')
        results.append(train_control(args.data_dir, prepared, args.output_dir/'control',
                                     initialization=args.initialization, epochs=args.epochs or 12,
                                     max_steps=args.max_steps, device=args.device))
    if args.recipe in ('full-elbo', 'all'):
        full_output = args.output_dir/'fullELBO30'
        results.append(train_full_elbo(args.data_dir, prepared, args.checkpoint or args.model_dir/'training/original.pt',
                                      full_output, seed=args.seed or 22, epochs=args.epochs or 30, max_steps=args.max_steps))
    if args.recipe in ('teacher', 'all'):
        checkpoint = (full_output/'model.pt') if args.recipe == 'all' else (args.checkpoint or args.model_dir/'training/fullELBO30.pt')
        seeds = SEEDS if args.recipe == 'all' or args.seed is None else (args.seed,)
        for seed in seeds:
            results.append(train_teacher(args.data_dir, prepared, checkpoint, args.output_dir/f'teacher{seed}',
                                         seed=seed, max_steps=args.max_steps))
    (args.output_dir/'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
