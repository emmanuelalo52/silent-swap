"""Reliance measures from session logs, per condition (and per scripted policy for simulated logs).

Reliance = how often participants' final answer matches the assistant (Yin et al., 2019 use the
same agreement measure). It is split by when and where it matters:
  before swap        agreement on questions 1-12 (all conditions see the FP16 build)
  after, stable      agreement after the swap on questions both builds answer the same way
  after, flipped     agreement after the swap on questions whose answer changed with the build
  over-reliance      final answer = assistant's answer when the assistant is wrong (after swap)
  switched to AI     initial answer differed from the assistant and the final answer adopted it

Usage: python3 src/analyze_sessions.py [--logs logs/sim] [--out analysis/session_summary_sim.md]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWAP_AT = json.loads((ROOT / "analysis" / "session_plan.json").read_text())["swap_at"]


def rate(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else None


def fmt(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def summarise(records):
    before = [r for r in records if r["n"] < SWAP_AT]
    after = [r for r in records if r["n"] >= SWAP_AT]
    after_flip = [r for r in after if r["flipped_item"]]
    after_stable = [r for r in after if not r["flipped_item"]]
    after_ai_wrong = [r for r in after if not r["ai_correct"]]
    return {
        "participants": len({r["pid"] for r in records}),
        "before swap": rate(before, "followed_ai"),
        "after, stable": rate(after_stable, "followed_ai"),
        "after, flipped": rate(after_flip, "followed_ai"),
        "over-reliance": rate(after_ai_wrong, "followed_ai"),
        "switched to AI": rate(after, "switched_to_ai"),
        "accuracy": rate(records, "final_correct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/sim")
    ap.add_argument("--out", default="analysis/session_summary_sim.md")
    args = ap.parse_args()
    records = []
    for path in sorted((ROOT / args.logs).glob("*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"no logs in {args.logs}")

    groups = defaultdict(list)
    for r in records:
        groups[(r["condition"], r.get("policy", "-"))].append(r)
    cols = ["participants", "before swap", "after, stable", "after, flipped", "over-reliance",
            "switched to AI", "accuracy"]
    simulated = any(r["pid"].startswith("SIM-") for r in records)
    lines = [("**SIMULATED participants: a pipeline test, not evidence about people.**\n" if simulated else ""),
             "Agreement = final answer matches the assistant's suggestion.", "",
             "| Condition | Policy | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for (cond, pol), rows in sorted(groups.items()):
        s = summarise(rows)
        lines.append(f"| {cond} | {pol} | {s['participants']} | " +
                     " | ".join(fmt(s[c]) for c in cols[1:]) + " |")
    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
