"""Participant bootstrap for fixed held-out predictions and their paired contrasts."""

import numpy as np


METRICS = ("AUC", "BAcc", "MCC", "Brier")
PAIRS = {
    "primary": (("Fusion", "GAP", 1), ("Fusion", "Attn", 1)),
    "handling": (("D1A", "FULL", 2), ("I2", "FULL", 2)),
    "core": (("SHRINK", "FULL", 4), ("SHRINK", "B_floor", 4), ("SHRINK", "D1A", 4),
             ("SHRINK", "I2", 4), ("FULL", "B_floor", 3), ("FULL", "D1A", 3), ("FULL", "I2", 3)),
    "targeted": tuple((name, "FULL", 4) for name in ("Binary_attention", "Equal_readout", "Binary_only", "No_anchor")),
    "quality": tuple((a, b, 9) for a, b in (
        ("FULL", "B_floor"), ("FULL", "Mean_q"), ("FULL", "Permuted_q"), ("FULL", "Readout_only"),
        ("FULL", "No_q_attention"), ("FULL", "Q_feature"), ("Q_feature", "Constant_feature"),
        ("Readout_only", "B_floor"), ("No_q_attention", "Readout_only"))),
}


def _primary_indices(rng, labels):
    n = len(labels)
    for _ in range(101):  # One initial draw and at most 100 retries.
        row = rng.integers(0, n, size=n, dtype=np.int64)
        positives = labels[row].sum()
        if 0 < positives < n:
            return row
    raise RuntimeError("Single-class bootstrap draw retry limit exhausted.")


def draw_counts(labels, protocol, draws):
    """Yield blocks of common participant multiplicities, preserving each protocol's RNG."""
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n = len(labels)
    seed = {"core": 2026091101, "quality": 20260908617}.get(protocol, 20260713)
    rng = np.random.default_rng(seed)
    if protocol == "targeted":
        # The study generated this integer array in one call. Splitting int16
        # draws across calls can discard a buffered value at a batch boundary.
        dtype = np.int16 if n - 1 <= np.iinfo(np.int16).max else np.int64
        targeted_indices = rng.integers(0, n, size=(draws, n), dtype=dtype)
    for start in range(0, draws, 128):
        size = min(128, draws - start)
        if protocol == "quality":
            counts = np.zeros((size, n), dtype=np.int64)
            for group in (positive, negative):
                counts[:, group] = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size)
        else:
            if protocol == "primary":
                index = [_primary_indices(rng, labels) for _ in range(size)]
            elif protocol == "core":
                index = [np.concatenate([rng.choice(g, len(g)) for g in (negative, positive)]) for _ in range(size)]
            elif protocol == "targeted":
                index = targeted_indices[start:start + size]
            else:
                index = rng.integers(0, n, size=(size, n), dtype=np.int64)
            counts = np.stack([np.bincount(row, minlength=n) for row in index])
        yield counts


def metrics(labels, scores, counts):
    """Return one (draw, metric) array. AUC uses half credit for ties."""
    positive = labels == 1
    wp, wn = counts[:, positive], counts[:, ~positive]
    p, n = wp.sum(1), wn.sum(1)
    pos, neg = scores[positive], scores[~positive]
    wins = (pos[:, None] > neg).astype(float) + 0.5 * (pos[:, None] == neg)
    auc = np.divide(np.einsum("bi,ij,bj->b", wp, wins, wn, optimize=True), p*n,
                    out=np.full(len(p), np.nan), where=p*n > 0)
    tp, fp = wp @ (pos >= 0.5), wn @ (neg >= 0.5)
    tn, fn = n - fp, p - tp
    sensitivity = np.divide(tp, p, out=np.zeros(len(p)), where=p > 0)
    specificity = np.divide(tn, n, out=np.zeros(len(n)), where=n > 0)
    represented_classes = (p > 0).astype(int) + (n > 0)
    bacc = (sensitivity + specificity) / represented_classes
    denomin = np.sqrt((tp+fp).astype(float) * p * n * (tn+fn))
    mcc = np.divide(tp*tn - fp*fn, denomin, out=np.zeros(len(p)), where=denomin > 0)
    brier = counts @ np.square(scores - labels) / counts.sum(1)
    return np.column_stack((auc, bacc, mcc, brier))


def compare_predictions(predictions, conditions, labels, protocol, draws=10000):
    labels = np.asarray(labels)
    if draws < 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Both labels and positive bootstrap draws are required.")
    if predictions.shape[-1] != len(labels) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be complete and aligned by participant.")
    if protocol == "primary":
        branches = predictions[0].mean(axis=0)
        scores = {"GAP": branches[:1], "Attn": branches[1:], "Fusion": branches.mean(0, keepdims=True)}
    else:
        scores = dict(zip(conditions, predictions.mean(axis=2)))

    def evaluate(weights):
        return {name: np.mean([metrics(labels, run, weights) for run in runs], axis=0)
                for name, runs in scores.items()}

    point = evaluate(np.ones((1, len(labels)), dtype=int))
    samples = {key: [] for key in scores}
    for counts in draw_counts(labels, protocol, draws):
        for key, values in evaluate(counts).items():
            samples[key].append(values)
    samples = {key: np.concatenate(values) for key, values in samples.items()}
    output = []

    def append(name, estimate, boot, family=1, contrast=False):
        for i, metric in enumerate(METRICS):
            if protocol in ("core", "quality") and metric != "AUC":
                continue
            count = family if metric == "AUC" else 1
            alpha = 0.05 / count
            values = boot[:, i]
            valid_values = values[np.isfinite(values)]
            limits = (np.quantile(valid_values, [0.025, 0.975, alpha/2, 1-alpha/2])
                      if len(valid_values) else np.full(4, np.nan))
            output.append(dict(comparison=name, paired=contrast, metric=metric,
                               point=float(estimate[0, i]), low95=float(limits[0]), high95=float(limits[1]),
                               family_confidence=1-alpha, family_low=float(limits[2]),
                               family_high=float(limits[3]), valid_draws=len(valid_values)))

    for name in scores:
        append(name, point[name], samples[name])
    for a, b, family in PAIRS[protocol]:
        if a in scores and b in scores:
            append(f"{a}-{b}", point[a]-point[b], samples[a]-samples[b], family, True)
    return output
