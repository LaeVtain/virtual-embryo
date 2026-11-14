"""Train the mature T2 control or the selected heart split-budget continuation."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..paths import task_assets
from . import losses
from .expression import training_rollout as expression_rollout
from .networks import ConditionalEndpoint, ContinuousNeighborhoodField
from .prepare import PAIRS
from .spatial import spatial_features, training_rollout as spatial_rollout


def vector_norm(gradients):
    return torch.sqrt(sum(gradient.detach().square().sum() for gradient in gradients))


def train(setting, prepared, output, *, mode="continue", steps=None, seed=1,
          model_dir=None, device="cpu"):
    """The continuation starts a fresh Adam optimizer from mature weights."""
    if mode not in ("control", "continue") or (mode == "continue" and setting != "heart"):
        raise ValueError("Use control for either setting, or continue for heart")
    steps = (300 if mode == "continue" else 3000) if steps is None else steps
    if steps < 1:
        raise ValueError("Training needs at least one optimizer step")
    folder = Path(model_dir) if model_dir is not None else task_assets("t2")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))[setting]
    times = config["times"][:3]
    data = joblib.load(prepared)
    stages = []
    for stage in data["stages"]:
        x, c = (stage["x"], stage["c"]) if isinstance(stage, dict) else stage
        stages.append((torch.as_tensor(x, device=device), torch.as_tensor(c, device=device)))
    n, genes = stages[0][0].shape
    if n < 128 or any(len(stage[0]) != n for stage in stages):
        raise ValueError("Training requires equally sized stages of at least 128 cells")
    torch.manual_seed(seed)
    expression = ConditionalEndpoint(genes).to(device)
    spatial = ContinuousNeighborhoodField(genes + 2).to(device)
    networks = torch.nn.ModuleList([expression, spatial])
    if mode == "continue":
        checkpoint = torch.load(folder / f"{setting}_control.pt", map_location=device, weights_only=True)
        expression.load_state_dict(checkpoint["expression"])
        spatial.load_state_dict(checkpoint["spatial"])
    optimizer = torch.optim.Adam(networks.parameters(), lr=1e-4 if mode == "continue" else 5e-4)
    parameters = list(networks.parameters())
    boundary = len(list(expression.parameters()))
    groups = (slice(0, boundary), slice(boundary, None))
    stats = {j: losses.tensors(value, device) for j, value in data["stats"].items()} if mode == "continue" else {}
    rng = np.random.default_rng(100 + seed)
    generator = torch.Generator().manual_seed(701 + seed)
    trace = hashlib.sha256()
    started = time.monotonic()
    initial_norms = None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as log:
        for step in range(steps):
            i, j = PAIRS[step % 3]
            a = rng.choice(n, 128, replace=False)
            cdf = np.cumsum(data["plans"][i, j][a], axis=1)
            cdf[:, -1] = 1
            b = (cdf < rng.random((128, 1))).sum(1)
            x, y = stages[i][0][a], stages[j][0][b]
            c, cy = stages[i][1][a], stages[j][1][b]
            c, cy = c - c.mean(0), cy - cy.mean(0)
            noise = torch.randn(x.shape, generator=generator).to(device)
            u = torch.rand((128, 1), generator=generator).to(device)
            dequantization = torch.rand(y.shape, generator=generator).to(device)
            target = torch.where(y == 0, -data["tau"] * dequantization, y)
            values = (a, b, noise.cpu().numpy(), u.cpu().numpy(), dequantization.cpu().numpy()) if mode == "continue" else (a, b)
            for value in values:
                trace.update(value.tobytes())
            duration = times[j] - times[i]
            features = spatial_features(x, times[i], times[j], times[0], times[2] - times[0])
            velocity = expression((1-u)*noise + u*target, u[:, 0], x,
                                  u[:, 0]*0 + times[i], u[:, 0]*0 + duration, times[0], times[2])
            expression_loss = (velocity - (target-noise)).square().mean()
            velocity = spatial((1-u[0, 0])*c + u[0, 0]*cy, features)
            spatial_loss = ((velocity-velocity.mean(0)) - (cy-c)/duration).square().mean()
            base = expression_loss + spatial_loss
            optimizer.zero_grad(set_to_none=True)
            record = dict(update=step+1, pair=[i, j], expression=float(expression_loss.detach()),
                          spatial=float(spatial_loss.detach()))
            if mode == "control":
                base.backward()
            else:
                predicted_expression = expression_rollout(expression, x, times[i], times[j], times[0], times[2], noise)
                predicted_coordinates = spatial_rollout(spatial, c, features, duration)
                auxiliary, _ = losses.balanced_objective(
                    losses.raw_terms(predicted_expression, stats[j], predicted_coordinates),
                    stats[j], (predicted_expression, predicted_coordinates))
                base_gradients = torch.autograd.grad(base, parameters)
                auxiliary_gradients = torch.autograd.grad(auxiliary, parameters)
                base_norms = [float(vector_norm(base_gradients[group])) for group in groups]
                auxiliary_norms = [float(vector_norm(auxiliary_gradients[group])) for group in groups]
                if initial_norms is None:
                    initial_norms = base_norms
                factors = [min(1., .25*max(norm, .1*initial)/max(auxiliary_norm, 1e-12))
                           for norm, initial, auxiliary_norm in zip(base_norms, initial_norms, auxiliary_norms)]
                for group, factor in zip(groups, factors):
                    for parameter, base_gradient, auxiliary_gradient in zip(
                            parameters[group], base_gradients[group], auxiliary_gradients[group]):
                        parameter.grad = base_gradient + factor*auxiliary_gradient
                record.update(auxiliary=float(auxiliary.detach()), factors=factors,
                              base_gradient_norms=base_norms, auxiliary_gradient_norms=auxiliary_norms)
            record["gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True))
            optimizer.step()
            log.write(json.dumps(record, allow_nan=False) + "\n")
            if step == 0 or (step+1) % 100 == 0:
                print(setting, mode, step+1, round(time.monotonic()-started, 1), flush=True)
    torch.save(dict(expression=expression.state_dict(), spatial=spatial.state_dict(),
                    seed=seed, updates=steps), output)
    report = dict(setting=setting, mode=mode, seed=seed, steps=steps,
                  schedule_hash=trace.hexdigest(), seconds=time.monotonic()-started,
                  checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("heart", "embryo"), required=True)
    parser.add_argument("--mode", choices=("control", "continue"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        train(args.setting, args.prepared, args.output, mode=args.mode, steps=args.steps,
              seed=args.seed, model_dir=args.model_dir, device=args.device)


if __name__ == "__main__":
    main()
