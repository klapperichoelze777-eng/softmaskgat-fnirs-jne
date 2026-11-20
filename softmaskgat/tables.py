"""Rebuild the published table cells from the three submitted aggregate CSVs."""

import csv
import io
from importlib.resources import files
from zipfile import ZipFile


def table_rows(supplement=None):
    names = ("r3/ensemble_interval_comparison.csv", "core_families.csv", "q_comparisons.csv")
    if supplement is None:
        package_data = files("softmaskgat").joinpath("aggregates")
        contents = [package_data.joinpath(n.split("/")[-1]).read_bytes() for n in names]
    else:
        with ZipFile(supplement) as z:
            contents = [z.read(n) for n in names]
    ensemble, core, quality = [
        list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")))) for data in contents]

    def ci(row, low, high, digits):
        return f"[{float(row[low]):.{digits}f}, {float(row[high]):.{digits}f}]"

    t2 = [["Variant", "AUC [95% CI]", "BAcc [95% CI]", "MCC [95% CI]", "Brier [95% CI]"]]
    by_key = {(r["model"], r["metric"]): r for r in ensemble}
    for key, label in (("V4", "V4 (binary mask)"), ("GAP", "GAP"), ("Attn", "Attn"), ("Fusion", "Fusion (50/50)")):
        cells = [label]
        for metric in ("AUC", "BAcc", "MCC", "Brier"):
            row = by_key[key, metric]
            cells.append(f"{float(row['point']):.3f} " + ci(row, "ensemble_ordinary_low", "ensemble_ordinary_high", 3))
        t2.append(cells)
    t11 = [["Contrast", "Mean ΔAUC", "Family-adjusted simultaneous CI"]]
    lookup = {r["statistic"]: r for r in core}
    for pair in ("FULL-B_floor", "FULL-D1A", "FULL-I2", "SHRINK-FULL", "SHRINK-B_floor", "SHRINK-D1A", "SHRINK-I2"):
        row = lookup["natural_auc_difference:" + pair]
        t11.append([pair, f"{float(row['point']):+.4f}", ci(row, "family_lower", "family_upper", 4)])
    t13 = [["Contrast", "Mean ΔAUC", "Descriptive 95% CI", "99.4444% simultaneous CI"]]
    lookup = {r["contrast"]: r for r in quality}
    for pair in ("FULL-B_floor", "FULL-No_q_attention", "FULL-Q_feature"):
        row = lookup[pair]
        t13.append([pair, f"{float(row['mean_repeat_paired_AUC_difference']):+.4f}",
                    ci(row, "descriptive_95_low", "descriptive_95_high", 4), ci(row, "adjusted_CI_low", "adjusted_CI_high", 4)])
    return {2: t2, 11: t11, 13: t13}
