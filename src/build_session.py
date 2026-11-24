"""Build the study's session plan from the recorded runs.

Decisions this script makes, each from the measured data rather than by hand:
  1. Which quantized build replaces FP16 at the swap: the configuration whose accuracy stays
     within --max-acc-change of FP16 (so benchmark accuracy is "matched") and that flips the
     most answers.
  2. Whether the swap needs a notice: only if its flip rate reaches --notice-threshold. The
     threshold sits above the FP32-vs-FP16 flip rate, which is ordinary numerical noise, so a
     notice is not raised for changes that are noise (the cry-wolf problem).
  3. Which questions participants see: 12 before the swap (all "stable", answered the same by
     both builds) and 12 after, including every flipped question from the chosen subjects.

Output: analysis/session_plan.json (read by src/server.py; contains the correct answers).
Usage: python3 src/build_session.py [--after auto|w8|fp8|...]
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mmlu_subset.json"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"

# General-knowledge subjects a non-specialist participant can engage with.
SUBJECTS = ["high_school_geography", "high_school_psychology", "marketing", "nutrition", "sociology",
            "us_foreign_policy", "world_religions", "miscellaneous", "management", "human_aging",
            "public_relations", "high_school_government_and_politics", "global_facts", "prehistory",
            "astronomy", "high_school_biology", "business_ethics"]
# Sensitive subjects (for example human_sexuality) are left out so the study needs no content warnings.
BUILD_NAMES = {"fp16": "standard 16-bit build", "fp32": "full-precision 32-bit build",
               "w8": "compressed 8-bit build", "fp8": "compressed 8-bit (FP8) build",
               "w4g128": "compressed 4-bit build", "w4g32": "compressed 4-bit build"}
PER_PHASE = 12


def load_run(name):
    run = json.loads((RUNS / f"{name}.json").read_text())
    return run["meta"], {r["id"]: r for r in run["results"]}


def readable(item):
    return len(item["question"]) <= 320 and all(len(c) <= 110 for c in item["choices"])


def one_in(rate):
    return f"about 1 in {round(1 / rate)}" if rate > 0 else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="fp16")
    ap.add_argument("--after", default="auto")
    ap.add_argument("--max-acc-change", type=float, default=0.02)
    ap.add_argument("--notice-threshold", type=float, default=0.02)
    ap.add_argument("--n-subjects", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    summary = json.loads((ANALYSIS / "flip_summary.json").read_text())
    rows = {r["config"]: r for r in summary["configs"]}
    if args.after == "auto":
        matched = [r for r in rows.values()
                   if r["config"].startswith(("w", "fp8")) and abs(r["delta_accuracy"]) <= args.max_acc_change]
        if not matched:
            raise SystemExit("no quantized config keeps accuracy within the limit")
        after = max(matched, key=lambda r: r["flip_rate"])["config"]
    else:
        after = args.after
    swap = rows[after]
    noise = rows.get("fp32", {}).get("flip_rate")

    items = {it["id"]: it for it in json.loads(DATA.read_text())}
    _, before_run = load_run(args.before)
    after_meta, after_run = load_run(after)

    # Pick subjects: readable questions, decent reference accuracy, and some flips to observe.
    stats = []
    for s in SUBJECTS:
        ids = [i for i, it in items.items() if it["subject"] == s and readable(it)]
        if len(ids) < 8:
            continue
        acc = sum(before_run[i]["correct"] for i in ids) / len(ids)
        flips = sum(before_run[i]["pred"] != after_run[i]["pred"] for i in ids)
        stats.append((s, ids, acc, flips))
    good = [x for x in stats if x[2] >= 0.5] or stats
    good.sort(key=lambda x: (x[3], x[2]), reverse=True)
    chosen = good[:args.n_subjects]

    flipped, stable = [], {}
    for s, ids, _, _ in chosen:
        for i in ids:
            (flipped if before_run[i]["pred"] != after_run[i]["pred"] else stable.setdefault(s, [])).append(i)
    # Right-to-wrong flips first: they test over-reliance, the main harm in the A1 gap.
    flipped.sort(key=lambda i: (not before_run[i]["correct"], i))
    flipped = flipped[:5]

    phase1 = []
    for s in stable:
        rng.shuffle(stable[s])
    while len(phase1) < PER_PHASE:
        for s in list(stable):
            if stable[s] and len(phase1) < PER_PHASE:
                phase1.append(stable[s].pop())
    rest = [i for s in stable for i in stable[s]]
    rng.shuffle(rest)
    phase2 = flipped + rest[:PER_PHASE - len(flipped)]
    rng.shuffle(phase2)

    def entry(n, i):
        b, a = before_run[i], after_run[i]
        it = items[i]
        return {"n": n, "id": i, "subject": it["subject"].replace("_", " "), "question": it["question"],
                "choices": it["choices"], "answer": it["answer"],
                "before": {"pred": b["pred"], "confidence": max(b["probs"]), "correct": b["correct"]},
                "after": {"pred": a["pred"], "confidence": max(a["probs"]), "correct": a["correct"]},
                "flipped": b["pred"] != a["pred"]}

    plan_items = [entry(n + 1, i) for n, i in enumerate(phase1 + phase2)]
    notify = swap["flip_rate"] >= args.notice_threshold
    plan = {
        "model": summary["model"],
        "before": {"config": args.before, "name": BUILD_NAMES[args.before],
                   "accuracy": summary["reference_accuracy"]},
        "after": {"config": after, "name": BUILD_NAMES[after], "label": after_meta["label"],
                  "accuracy": swap["accuracy"], "flip_rate": swap["flip_rate"]},
        "swap_at": PER_PHASE + 1,
        "notice_rule": {"threshold": args.notice_threshold, "flip_rate": swap["flip_rate"],
                        "noise_floor_fp32_vs_fp16": noise, "notify": notify},
        "notice": {
            "title": f"Assistant update from question {PER_PHASE + 1}",
            "changed": f"The assistant now runs a {BUILD_NAMES[after]} of the same model, which is cheaper to serve.",
            "unchanged": (f"Its overall accuracy on our test set is about the same: "
                          f"{100 * summary['reference_accuracy']:.1f}% before, {100 * swap['accuracy']:.1f}% after."),
            "watch": (f"But {one_in(swap['flip_rate'])} of its answers changed, mostly ones it was unsure about. "
                      f"Look twice at low-confidence suggestions."),
        } if notify else None,
        "subjects": [s.replace("_", " ") for s, *_ in chosen],
        "items": plan_items,
    }
    (ANALYSIS / "session_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    n_flip = sum(e["flipped"] for e in plan_items)
    acc1 = sum(e["before"]["correct"] for e in plan_items[:PER_PHASE]) / PER_PHASE
    print(f"after={after} flip_rate={swap['flip_rate']:.4f} (noise floor {noise}) notify={notify}")
    print(f"subjects={plan['subjects']} flipped items in session={n_flip} assistant phase-1 accuracy={acc1:.2f}")


if __name__ == "__main__":
    main()
