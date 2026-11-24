"""Scripted participants that run through every condition, to test the pipeline end to end.

These are NOT findings about people. They check that sessions, routing, logging and analysis
work together and that the analysis can tell apart the behaviours it is meant to measure.
Three scripted policies:
  trusting    follows the assistant whenever its confidence is at least 40%
  calibrated  follows it only in subjects where it has been right at least half the time so far
  cautious    like calibrated, but after an update notice also needs 60% confidence

Writes logs/sim/<pid>.jsonl using the same record format as real sessions (src/server.py).
Usage: python3 src/simulate_participants.py [--per-cell 10]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import CONDITIONS, check_answer, load_plan, session_for  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "sim"
POLICIES = ("trusting", "calibrated", "cautious")


def own_answer(item, skill, rng):
    if rng.random() < skill:
        return item["answer"]
    return rng.choice([l for l in "ABCD" if l != item["answer"]])


def run_one(plan, condition, policy, pid, rng, skill=0.45):
    view = session_for(plan, condition)
    record_by_subject, records = {}, []
    for it_plan, it in zip(plan["items"], view["items"]):
        initial = own_answer(it_plan, skill, rng)
        ai, conf = it["ai"]["letter"], it["ai"]["confidence"]
        r, n = record_by_subject.get(it["subject"], (0, 0))
        track = r / n if n else 0.5
        if policy == "trusting":
            follow = conf >= 0.4
        else:
            follow = track >= 0.5 and conf >= 0.4
            if policy == "cautious" and view["notice"] and it["n"] >= view["swap_at"]:
                follow = follow and conf >= 0.6
        final = ai if follow else initial
        rec, fb = check_answer(plan, {"pid": pid, "condition": condition, "n": it["n"],
                                      "initial": initial, "final": final})
        rec["policy"] = policy
        records.append(rec)
        record_by_subject[it["subject"]] = (r + fb["ai_correct"], n + 1)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    plan = load_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jsonl"):
        old.unlink()
    n = 0
    for condition in CONDITIONS:
        for policy in POLICIES:
            for k in range(args.per_cell):
                pid = f"SIM-{condition}-{policy}-{k:02d}"
                recs = run_one(plan, condition, policy, pid, rng)
                (OUT / f"{pid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
                n += 1
    print(f"wrote {n} simulated sessions to {OUT}")


if __name__ == "__main__":
    main()
