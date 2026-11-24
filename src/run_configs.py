"""Answer every question in data/mmlu_subset.json under several serving configurations.

Each configuration is the same Qwen3-0.6B weights served differently: full FP32, FP16, or with
the decoder's linear-layer weights quantized (weight-only fake quantization: weights are rounded
to the low-precision grid and stored back as FP16, which reproduces the numerics of W8/FP8/W4
serving without needing special kernels).

Answers are scored, not sampled: one forward pass per question, then a softmax over the logits of
the four answer letters. This removes sampling randomness, so any difference between two
configurations comes from the serving change itself (see He, 2025, on nondeterminism).

Output: runs/<config>.json with, per question, the predicted letter, the four letter
probabilities, the top-two margin, and whether the prediction is correct.

Usage: python3 src/run_configs.py [--configs fp16 w8 fp8 w4g128] [--model PATH]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mmlu_subset.json"
RUNS = ROOT / "runs"
DEFAULT_MODEL = str(Path.home() / "models" / "Qwen3-0.6B")
LETTERS = ["A", "B", "C", "D"]

CONFIGS = {
    "fp32":   {"dtype": torch.float32, "quant": None, "label": "FP32 weights and activations"},
    "fp16":   {"dtype": torch.float16, "quant": None, "label": "FP16 weights and activations"},
    "w8":     {"dtype": torch.float16, "quant": ("int", 8, None), "label": "INT8 weights, per-channel RTN"},
    "fp8":    {"dtype": torch.float16, "quant": ("fp8", 8, None), "label": "FP8 (E4M3) weights, per-channel scale"},
    "w4g128": {"dtype": torch.float16, "quant": ("int", 4, 128), "label": "INT4 weights, RTN, group size 128"},
    "w4g32":  {"dtype": torch.float16, "quant": ("int", 4, 32), "label": "INT4 weights, RTN, group size 32"},
}


def prompt_for(item):
    subject = item["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", item["question"]]
    lines += [f"{l}. {c}" for l, c in zip(LETTERS, item["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def fake_quantize_(weight, kind, bits, group):
    """Round a [out, in] weight matrix to a low-precision grid in place (symmetric, per channel or group)."""
    w = weight.float()
    out_f, in_f = w.shape
    g = group or in_f
    w = w.reshape(out_f, in_f // g, g)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    if kind == "int":
        qmax = 2 ** (bits - 1) - 1
        scale = amax / qmax
        w = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    else:  # fp8 e4m3: scale each channel so its max maps to 448, cast down and back
        scale = amax / 448.0
        w = (w / scale).to(torch.float8_e4m3fn).float() * scale
    weight.copy_(w.reshape(out_f, in_f).to(weight.dtype))


def quantize_model_(model, kind, bits, group):
    n = 0
    for layer in model.model.layers:
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                fake_quantize_(module.weight.data, kind, bits, group)
                n += 1
    return n


@torch.no_grad()
def score(model, tok, items, batch_size, device):
    letter_ids = [tok.encode(" " + l, add_special_tokens=False)[0] for l in LETTERS]
    prompts = [prompt_for(it) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    results = [None] * len(items)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        probs = torch.softmax(logits[:, letter_ids].float(), dim=-1).cpu()
        for row, i in zip(probs, idx):
            p = row.tolist()
            top = sorted(p, reverse=True)
            pred = LETTERS[max(range(4), key=lambda k: p[k])]
            results[i] = {"id": items[i]["id"], "pred": pred, "probs": [round(x, 5) for x in p],
                          "margin": round(top[0] - top[1], 5), "correct": pred == items[i]["answer"]}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    items = json.loads(DATA.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    RUNS.mkdir(exist_ok=True)

    for name in args.configs:
        cfg = CONFIGS[name]
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=cfg["dtype"]).to(device).eval()
        n_quant = quantize_model_(model, *cfg["quant"]) if cfg["quant"] else 0
        results = score(model, tok, items, args.batch_size, device)
        acc = sum(r["correct"] for r in results) / len(results)
        meta = {"config": name, "label": cfg["label"], "model": Path(args.model).name,
                "quantized_linear_layers": n_quant, "n_items": len(results),
                "accuracy": round(acc, 4), "seconds": round(time.time() - t0, 1),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
        (RUNS / f"{name}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"{name:7s} acc={acc:.4f} quantized_layers={n_quant} time={meta['seconds']}s")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
