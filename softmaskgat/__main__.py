"""Command-line interface for the synthetic example, study protocols and tables."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="softmaskgat")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Train and evaluate on independently simulated signals.")
    demo.add_argument("--participants", type=int, default=8)
    run = commands.add_parser("run", help="Use an input NPZ with one row per participant.")
    run.add_argument("input", type=Path)
    for sub in (demo, run):
        sub.add_argument("--protocol", choices=("primary", "handling", "core", "targeted", "quality"), default="primary")
        sub.add_argument("--conditions", nargs="+", help="Select a subset of the protocol's conditions.")
        sub.add_argument("--seeds", nargs="+", type=int)
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--threads", type=int, default=2, help="Number of PyTorch CPU threads.")
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--folds", type=int, help="Stratified fold count for core and quality.")
        sub.add_argument("--draws", type=int)
        sub.add_argument("--output", type=Path, help="New output directory for held-out predictions and statistics.")
    tbl = commands.add_parser("tables", help="Generate tables 2, 11 and 13 from aggregate values.")
    tbl.add_argument("--supplement", type=Path, help="Original supplementary ZIP (optional).")
    tbl.add_argument("--output", type=Path, help="New directory for the CSV tables.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("The output directory must be new.")
    if args.command == "tables":
        from .tables import table_rows
        tables = table_rows(args.supplement)
        if args.output:
            args.output.mkdir(parents=True)
        for key, rows in tables.items():
            if args.output:
                with (args.output / f"table_{key}.csv").open("w", newline="", encoding="utf-8") as fp:
                    csv.writer(fp).writerows(rows)
            else:
                print(f"Table {key}")
                for row in rows:
                    print(" | ".join(row))
        return

    import numpy as np
    import torch
    from .data import Dataset, synthetic
    from .statistics import compare_predictions
    from .training import cross_validate

    is_demo = args.command == "demo"
    epochs = args.epochs if args.epochs is not None else (1 if is_demo else 120)
    folds = args.folds if args.folds is not None else (2 if is_demo else 10)
    draws = args.draws if args.draws is not None else (128 if is_demo else 10000)
    if args.threads < 1 or draws < 1:
        parser.error("threads and draws must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this host.")
    torch.set_num_threads(args.threads)
    dataset = synthetic(args.participants) if is_demo else Dataset.load(args.input)
    seeds = (42,) if is_demo and args.seeds is None else args.seeds
    predictions, conditions, seeds, branches = cross_validate(
        dataset, args.protocol, args.conditions, epochs=epochs, folds=folds, seeds=seeds, device=args.device)
    stats = compare_predictions(predictions, conditions, dataset.label, args.protocol, draws)
    settings = {"input": "synthetic" if is_demo else "user supplied", "protocol": args.protocol,
                "epochs": epochs, "folds": folds if args.protocol in ("core", "quality") else len(dataset.label),
                "draws": draws, "conditions": conditions, "seeds": seeds, "branches": branches,
                "oof_shape": list(predictions.shape), "device": args.device}
    if args.output:
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output / "predictions.npz", probabilities=predictions, labels=dataset.label,
                            ids=dataset.ids, conditions=conditions, seeds=seeds, branches=branches)
        (args.output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        with (args.output / "statistics.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)
    print(json.dumps(settings))
    for row in stats:
        if row["metric"] == "AUC":
            print(f"{row['comparison']:28} AUC{' difference' if row['paired'] else ''} {row['point']:+.4f}")


if __name__ == "__main__":
    main()
