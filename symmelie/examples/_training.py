"""Shared CLI and training utilities; no optional dependencies or downloads."""
import argparse
import json
from pathlib import Path
import torch


def fit(model, sample, steps, *, lr=.01):
    """Train on fresh batches; report MSE on one fixed, unseen evaluation batch."""
    evaluation = sample(128)
    with torch.no_grad():
        baseline = torch.nn.functional.mse_loss(model(*evaluation[:-1]), evaluation[-1]).item()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = sample(32)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(*batch[:-1]), batch[-1])
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a nonfinite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(*evaluation[:-1]), evaluation[-1]).item()
    return {"initial_test_mse": baseline, "final_test_mse": final}


def cli(run, *, steps=200):
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON metrics file")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    # Tiny dense examples are faster without large CPU thread pools.
    torch.set_num_threads(1)
    metrics = {"seed": args.seed, "steps": args.steps, **run(args.steps, args.seed)}
    report = json.dumps(metrics, indent=2, allow_nan=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
