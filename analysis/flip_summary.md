Reference: `fp16` (FP16 weights and activations), accuracy 42.0% on 1140 MMLU questions.

| Config | Weights | Accuracy | Change | Flips | Right to wrong | Wrong to right | Mean KL |
|---|---|---|---|---|---|---|---|
| fp32 | FP32 weights and activations | 42.2% | +0.2 pts | 4 (0.4%) | 1 | 3 | 3e-05 |
| w8 | INT8 weights, per-channel RTN | 42.8% | +0.8 pts | 82 (7.2%) | 22 | 31 | 0.01153 |
| fp8 | FP8 (E4M3) weights, per-channel scale | 42.9% | +0.9 pts | 132 (11.6%) | 40 | 50 | 0.04372 |
| w4g128 | INT4 weights, RTN, group size 128 | 25.9% | -16.1 pts | 500 (43.9%) | 260 | 76 | 0.6577 |
| w4g32 | INT4 weights, RTN, group size 32 | 34.1% | -7.9 pts | 693 (60.8%) | 255 | 165 | 0.87836 |
