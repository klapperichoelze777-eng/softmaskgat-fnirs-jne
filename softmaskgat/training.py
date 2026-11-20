"""Participant-level cross-validation for the SoftMaskGAT comparisons."""

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from torch.nn import functional as F

from .data import adjacency, interpolate, quality_weights, scale_anchor, spectral_ratios
from .model import ModelV5_SoftMask, ModelV6_RealSTGAT
from .routing import condition_q, forward_ragged, routed_forward, routes, valid_mappings
from .schedule import CosineWarmupScheduler
from .shrink import ProjectedAdamW, ShrinkAdapter, optimization_step


CONDITIONS = {
    "primary": ("FULL",),
    "handling": ("FULL", "D1A", "I2"),
    "core": ("FULL", "B_floor", "D1A", "I2", "SHRINK"),
    "targeted": ("FULL", "Binary_attention", "Equal_readout", "Binary_only", "No_anchor"),
    "quality": ("FULL", "B_floor", "Mean_q", "Permuted_q", "Readout_only",
                "No_q_attention", "Q_feature", "Constant_feature"),
}
BRANCHES = ("GAP", "Attn")
STUDY_SEEDS = (42, 123, 2026)


def make_model(branch, condition="FULL"):
    settings = {
        "Binary_attention": {"quality_combine": "binary"},
        "Equal_readout": {"readout_use_quality": False},
        "Binary_only": {"quality_combine": "binary", "readout_use_quality": False,
                        "anchor_use_quality": False},
        "No_anchor": {"residual_scale": 0.0},
    }.get(condition, {})
    if branch == "GAP":
        model = ModelV5_SoftMask(**settings)
    elif branch == "Attn":
        model = ModelV6_RealSTGAT(temporal_encoder_type="cnn_attn", **settings)
    else:
        raise ValueError(f"Unknown encoder: {branch}")
    if condition in ("Q_feature", "Constant_feature"):
        model.q_feature_adapter = torch.nn.Linear(1, 64, bias=True)
    if condition == "SHRINK":
        return ShrinkAdapter(model)
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_from(parts):
    return int(np.random.SeedSequence(parts).generate_state(1)[0])


def splits_for(label, protocol, seed, folds):
    if protocol not in ("core", "quality"):
        return list(LeaveOneOut().split(label))
    if folds < 2 or np.bincount(label, minlength=2).min() < folds:
        raise ValueError(f"Stratified {folds}-fold evaluation needs at least {folds} participants in each class.")
    split_seed = 42 if protocol == "core" else seed
    random_state = seed_from([20260908, split_seed, 90210])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return list(cv.split(np.zeros(len(label)), label))


@dataclass
class Fold:
    signal: torch.Tensor
    anchor: torch.Tensor
    mask: torch.Tensor
    q: torch.Tensor
    adj: torch.Tensor
    label: torch.Tensor
    condition: str

    def predict_logits(self, model, indices):
        s, a, m = self.signal[indices], self.anchor[indices], self.mask[indices]
        if self.condition == "D1A":
            signals = [row[:, keep] for row, keep in zip(s, m)]
            anchors = [row[keep] for row, keep in zip(a, m)]
            graphs = [self.adj[keep][:, keep] for keep in m]
            out = forward_ragged(model, signals, graphs, anchors)
        elif self.condition == "I2":
            out = model(s, self.adj, m, a, quality=None)
        elif self.condition in ("B_floor", "Mean_q", "Permuted_q", "Readout_only",
                               "No_q_attention", "Q_feature", "Constant_feature"):
            weights = routes(self.q[indices], m, self.condition)
            out = routed_forward(model, s, self.adj, m, a, *weights)
        else:
            out = model(s, self.adj, m, a, quality=self.q[indices])
        return out.squeeze(-1)


def train_fold(model, fold, train, test, epochs, namespace, seed, block):
    device = fold.signal.device
    train = torch.as_tensor(train, device=device)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if isinstance(model, ShrinkAdapter):
        optimizer = ProjectedAdamW(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = CosineWarmupScheduler(optimizer, 5e-4, min(5, epochs // 10), epochs, 5e-6)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        scheduler.step(epoch)
        if namespace is None:
            order = torch.randperm(len(train), device=device)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([namespace, seed, block, epoch, 801]))
            order = torch.as_tensor(rng.permutation(len(train)), device=device)
        losses = []
        for start in range(0, len(train), 16):
            rows = train[order[start:start + 16]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = fold.predict_logits(model, rows)
                targets = 0.95 * fold.label[rows] + 0.025
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            optimization_step(loss, model, optimizer, amp)
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss - 1e-4:
            best_loss, stale = mean_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(test), 16):
            batch = test[start:start + 16]
            predictions.extend(fold.predict_logits(model, batch).sigmoid().cpu().tolist())
    return predictions


def cross_validate(data, protocol, conditions=None, *, epochs=120, folds=10,
                   seeds=None, device="cpu"):
    """Return OOF probabilities as (condition, run, branch, participant)."""
    device = torch.device(device)
    if protocol not in CONDITIONS or not 1 <= epochs <= 120:
        raise ValueError("Choose a published protocol and an epoch limit in [1, 120].")
    conditions = tuple(conditions) if conditions is not None else CONDITIONS[protocol]
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS[protocol]):
        raise ValueError(f"Choose distinct conditions from {CONDITIONS[protocol]}.")
    study_seeds = (42,) if protocol in ("handling", "targeted") else STUDY_SEEDS
    seeds = tuple(seeds) if seeds is not None else study_seeds
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(study_seeds):
        raise ValueError(f"Allowed seeds for {protocol}: {study_seeds}")
    branches = ("GAP",) if protocol == "targeted" else BRANCHES
    namespace = {"core": 20260911, "quality": 20260908}.get(protocol)
    ratio_dtype = np.float32 if protocol == "primary" else np.float64
    ratios = spectral_ratios(data.signal, data.mask, dtype=ratio_dtype)
    rows = np.arange(len(data.label))
    cohort_q = quality_weights(ratios, data.mask, rows) if protocol == "primary" else None
    maps = valid_mappings(data.mask) if "Permuted_q" in conditions else None
    repaired = interpolate(data) if "I2" in conditions else None
    graph = torch.as_tensor(adjacency(data.coords), device=device)
    label = torch.as_tensor(data.label, device=device, dtype=torch.float32)
    predictions = np.full((len(conditions), len(seeds), len(branches), len(rows)), np.nan)
    for ci, condition in enumerate(conditions):
        values, prior = repaired if condition == "I2" else (data.signal, data.anchor)
        observed = np.ones_like(data.mask) if condition == "I2" else data.mask
        signal = torch.as_tensor(values, device=device)
        mask = torch.as_tensor(observed, device=device)
        for ri, seed in enumerate(seeds):
            split = splits_for(data.label, protocol, seed, folds)
            for bi, branch in enumerate(branches):
                seed_all(seed)
                for fi, (train, test) in enumerate(split):
                    block = STUDY_SEEDS.index(seed) * folds + fi if protocol == "quality" else fi
                    q = cohort_q if cohort_q is not None else quality_weights(ratios, data.mask, train)
                    if condition in ("Mean_q", "Permuted_q"):
                        variant = f"Permuted_q_{STUDY_SEEDS.index(seed)+1}" if condition == "Permuted_q" else condition
                        q = condition_q(q, data.mask, variant, maps)
                    original_rule = protocol in ("primary", "targeted", "handling") and condition not in ("I2", "D1A")
                    scaled = scale_anchor(prior, observed, train, original_rule)
                    fold = Fold(signal, torch.as_tensor(scaled, device=device), mask,
                                torch.as_tensor(q, device=device), graph, label, condition)
                    if namespace is not None:
                        seed_all(seed_from([namespace, seed, block, 19]))
                    model = make_model(branch, condition).to(device)
                    if namespace is not None:
                        seed_all(seed_from([namespace, seed, block, 23 if branch == "GAP" else 24]))
                    predictions[ci, ri, bi, test] = train_fold(
                        model, fold, train, test, epochs, namespace, seed, block)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Incomplete or non-finite held-out probabilities.")
    return predictions, conditions, seeds, branches
