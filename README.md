# scdmn-cifar

**Scene-Conditional Dynamic Mask Networks — CIFAR-10 / CIFAR-10-C validation.**

> ⚠️ This is a **demo / toy validation** repo. The goal is to check whether the
> core SCDMN hypothesis holds in a controlled setting, not to reach SOTA.
> The full driving-scale implementation lives elsewhere.

> *"Inspired by dissociative identity disorder — a single network that, depending
> on the scene it sees, routes its computation through a different subset of its
> own channels, as if switching between specialized 'identities' for day, rain,
> and night."*

---

## TL;DR

End-to-end driving models learn one weight matrix `W` that simultaneously
handles clear days, rain, darkness, and more. Gradients from different domains
compete on the same parameters, and causal confusion leaks across contexts.

**SCDMN** adds a tiny mask generator that reads a context embedding `z` from
the scene and produces a **channel-wise binary mask** for each stage. A
learnable per-stage gate `g_i ∈ [0,1]` decides how strongly that mask is
applied:

```python
out_i = layer_i(x) * (1 - g_i + g_i * M(z))
#   g_i → 0 : fully shared (low-level features, e.g. edges)
#   g_i → 1 : fully context-specific (high-level, scene-conditioned)
```

Parameter count stays at **~1× a plain ResNet18**, while different scenes
effectively run through different sub-networks that share the low-level trunk.

---

## Hypothesis under test

If the design works, we should see three things simultaneously on this toy setup:

1. **Gate pattern.** `g_i` rises monotonically with depth:
   ```
   stage 1 (64 ch)   g ≈ 0.05 – 0.20   (shared)
   stage 2 (128 ch)  g ≈ 0.20 – 0.40
   stage 3 (256 ch)  g ≈ 0.40 – 0.70
   stage 4 (512 ch)  g ≈ 0.70 – 0.95   (context-specific)
   ```
2. **Linear probe.** Per-stage features become more context-separable with
   depth (probe accuracy on context label goes up).
3. **Mask IoU.** At deep stages, different contexts route through largely
   **different channels** (low off-diagonal IoU).

If any one of these disagrees with the others, we have a failure mode to
investigate — that's the whole point of running the analysis, not just the
accuracy table.

---

## How it compares

| | DriveMoE (2025) | CausalVAD (2025) | **SCDMN (this)** |
|---|---|---|---|
| Added params | ~7× (expert FFNs) | small | **~1.03×** |
| Context-specialized? | mixed (load imbalance) | no | **yes, by design** |
| Where is separation? | extra FFN experts | query-level, post-hoc | **channel-level, inside W** |
| Weather/lighting-aware? | no | no | **yes** |

---

## Environment

WSL2 / Ubuntu 22.04 / Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scikit-learn matplotlib
```

---

## Data

**Option A — Official CIFAR-10-C (recommended for the final numbers).**

Download `brightness.npy`, `frost.npy`, `gaussian_noise.npy`, and `labels.npy`
from the CIFAR-10-C release (Hendrycks & Dietterich, 2019), and place them in:

```
data_cache/CIFAR-10-C/
├── brightness.npy
├── frost.npy
├── gaussian_noise.npy
└── labels.npy
```

Plain CIFAR-10 is downloaded automatically by `torchvision`.

**Option B — Synthetic fallback.** Pass `--no_official_c` and the code applies
simplified brightness / haze / Gaussian-noise corruptions in-process. Useful
for iteration, not for paper numbers.

---

## Smoke test (a few minutes on a single GPU)

```bash
python -m experiments.run_all --quick --no_official_c
```

Trains all three models for 2 epochs on 1000 samples per context. Accuracies
will be low (0.2–0.4) — this run only checks that shapes, masks, and the
backward pass work end-to-end.

---

## Full run

```bash
# Oracle context label: upper bound for "does the mask structure itself help?"
python -m experiments.run_all --epochs 30 --mode oracle

# CNN context encoder: the real setting (context inferred from the image)
python -m experiments.run_all --epochs 30 --mode cnn
```

Running **both** `oracle` and `cnn` is intentional. It separates two sub-questions:

- *Does the channel-mask structure help, assuming context is known?* → `oracle`
- *Can we also infer context from the raw image?* → `cnn`

If `oracle` already fails to beat the baseline, there's no point tuning `cnn`.

---

## What gets produced

Per run, under `experiments/runs/<run_name>_*`:

- `*_best.pt`, `*_final.pt` — checkpoints
- `*.log` — per-epoch training log, including gate values for SCDMN
- `<scdmn_run>_analysis.json` — gate values, per-stage linear-probe accuracies
  (context & class), and per-stage mask-IoU matrices
- `<scdmn_run>_analysis.png` — a 3-panel figure:
  1. Gate values per stage
  2. Linear-probe accuracy (context vs class) vs stage
  3. Mask IoU heatmap at the deepest stage

---

## Expected result shape

| Model | Params | overall | clean | brightness | frost | gaussian |
|---|---|---|---|---|---|---|
| Baseline (single ResNet18) | ~11M | baseline | … | … | … | … |
| Independent experts (4× ResNet18) | ~44M | upper bound | … | … | … | … |
| **SCDMN (cnn)** | **~11.3M** | **close to independent** | … | … | … | … |

SCDMN lands close to "independent experts" at ~25% of their parameters → the
mask routing is genuinely doing the work the experts used to require.

---

## Repo layout

```
scdmn-cifar/
├── models/
│   ├── scdmn_components.py     # Top-k STE, ContextEncoder (CNN / Oracle), MaskGenerator, GatedMaskApply
│   ├── scdmn_resnet.py         # SCDMN-ResNet18 (4 masked stages)
│   └── resnet_baseline.py      # Plain ResNet18 + IndependentExperts (oracle routing)
├── data/
│   └── multi_context_cifar.py  # CIFAR-10 + CIFAR-10-C multi-context dataset
├── experiments/
│   ├── trainer.py              # Unified training loop for all three variants
│   └── run_all.py              # Main entry point
└── utils/
    └── analysis.py             # Linear Probe, mask IoU, gate visualization
```

---

## Design decisions, for future-me

- **One (gate, mask) per stage — 4 total — not per BasicBlock.** Channel counts
  only change between stages, and stage-level granularity is enough to observe
  the shallow→deep hypothesis. Per-block masks would blow up the mask
  generator for little extra insight at this scale.
- **Gate initialized at `sigmoid(-2) ≈ 0.12`.** Training starts near-shared;
  the model has to actively *raise* `g_i` where context separation helps. This
  is what makes the "rising with depth" pattern meaningful instead of
  tautological.
- **Top-k STE for the mask.** Forward keeps the top-k channels (hard binary);
  backward flows gradient through the sigmoid scores unchanged. No explicit
  weight-gradient hook is needed — forward activation masking already
  zeroes the relevant weight gradients via chain rule.
- **Mask rescaled by `1 / sparsity` (= 2.0 at 50%).** Keeps the expected
  activation magnitude roughly constant regardless of sparsity, so the
  baseline and SCDMN operate at comparable feature scales.
- **Soft-mask warmup for the first 2 epochs** (`warmup_soft_mask_epochs`),
  then switch to hard Top-k STE. Hard binarization from step 0 destabilizes
  early learning.

---

## Where this goes next

If the gate-depth pattern and the linear-probe curve both show what the
hypothesis predicts on this toy benchmark, the next steps are:

1. Scale contexts (7–10 CIFAR-10-C corruptions, or BDD100K condition splits).
2. Replace the classification head with a perception head on a driving dataset.
3. Port to the Orin Nano E2E vehicle stack with TensorRT deployment.

Each of those is a separate repo. This one stays as the canonical
controlled-setting check.

---

## License / attribution

For internal research use. Inspired in spirit by work on modular networks,
conditional computation, and mixture-of-experts — but the specific mechanism
(per-stage learnable gate × scene-conditioned channel mask, inside a single
weight matrix) is developed here.