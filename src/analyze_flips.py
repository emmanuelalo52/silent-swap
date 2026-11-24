"""Compare every serving configuration with the FP16 reference: accuracy, flips and where flips occur.

A flip is a question whose predicted answer differs between the reference and another
configuration (Dutta et al., 2024). Right-to-wrong and wrong-to-right flips are counted
separately, and flips are broken down by the reference's top-two margin (how sure FP16 was).

Output: analysis/flip_summary.json, analysis/flip_summary.md, analysis/flips_by_margin.png
Usage: python3 src/analyze_flips.py [--reference fp16]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "analysis"
MARGIN_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]


def load(name):
    return json.loads((RUNS / f"{name}.json").read_text())


def kl(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-9)) for pi, qi in zip(p, q) if pi > 0)


def compare(ref, other):
    pairs = list(zip(ref["results"], other["results"]))
    assert all(a["id"] == b["id"] for a, b in pairs)
    flips = [(a, b) for a, b in pairs if a["pred"] != b["pred"]]
    by_bin = []
    for lo, hi in MARGIN_BINS:
        in_bin = [(a, b) for a, b in pairs if lo <= a["margin"] < hi]
        n_flip = sum(a["pred"] != b["pred"] for a, b in in_bin)
        by_bin.append({"margin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(in_bin), "flips": n_flip,
                       "flip_rate": round(n_flip / len(in_bin), 4) if in_bin else None})
    return {
        "config": other["meta"]["config"],
        "label": other["meta"]["label"],
        "accuracy": other["meta"]["accuracy"],
        "delta_accuracy": round(other["meta"]["accuracy"] - ref["meta"]["accuracy"], 4),
        "flips": len(flips),
        "flip_rate": round(len(flips) / len(pairs), 4),
        "right_to_wrong": sum(a["correct"] and not b["correct"] for a, b in flips),
        "wrong_to_right": sum(b["correct"] and not a["correct"] for a, b in flips),
        "mean_kl": round(sum(kl(a["probs"], b["probs"]) for a, b in pairs) / len(pairs), 5),
        "flips_by_reference_margin": by_bin,
    }


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["flips"] > 0]
    labels = [b["margin"] for b in rows[0]["flips_by_reference_margin"]]
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    width = 0.8 / len(rows)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for k, r in enumerate(rows):
        xs = [i + (k - (len(rows) - 1) / 2) * width for i in range(len(labels))]
        ys = [100 * (b["flip_rate"] or 0) for b in r["flips_by_reference_margin"]]
        ax.bar(xs, ys, width * 0.9, label=f'{r["config"]} ({100 * r["flip_rate"]:.1f}% overall)',
               color=colors[k % len(colors)])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("FP16 top-two margin (how sure the reference answer was)")
    ax.set_ylabel("Answers that flip (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e2dc", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="fp16")
    args = ap.parse_args()

    ref = load(args.reference)
    names = sorted(p.stem for p in RUNS.glob("*.json") if p.stem != args.reference)
    rows = [compare(ref, load(n)) for n in names]
    rows.sort(key=lambda r: r["flip_rate"])
    summary = {"reference": args.reference, "reference_accuracy": ref["meta"]["accuracy"],
               "n_items": ref["meta"]["n_items"], "model": ref["meta"]["model"], "configs": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "flip_summary.json").write_text(json.dumps(summary, indent=1))

    md = [f"Reference: `{args.reference}` ({ref['meta']['label']}), accuracy "
          f"{100 * ref['meta']['accuracy']:.1f}% on {ref['meta']['n_items']} MMLU questions.", "",
          "| Config | Weights | Accuracy | Change | Flips | Right to wrong | Wrong to right | Mean KL |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['label']} | {100 * r['accuracy']:.1f}% | "
                  f"{100 * r['delta_accuracy']:+.1f} pts | {r['flips']} ({100 * r['flip_rate']:.1f}%) | "
                  f"{r['right_to_wrong']} | {r['wrong_to_right']} | {r['mean_kl']} |")
    (OUT / "flip_summary.md").write_text("\n".join(md) + "\n")
    plot(rows, OUT / "flips_by_margin.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
