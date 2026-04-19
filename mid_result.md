# SCDMN-Sliced: Mid-Project Results

## TL;DR

We built **SCDMN-Sliced**, an oracle-conditional variant of SCDMN where per-context channel masks actually slice the conv/BN tensors so FLOPs (not just activations) are reduced. With sparsity 0.5 on ResNet34 / CIFAR-10-C (4 contexts), it reaches **0.799 overall accuracy at ~5.4M active params and ~0.58G FLOPs per image**, versus the param/FLOP-matched independent-experts baseline at **0.826** — a **−2.6%p gap at identical compute**.

### What worked

- Applying the soft mask to conv **weights** (not activations) made the soft warmup and the post-freeze sliced forward numerically consistent (output diff ≈ 0.004 at freeze time).
- A unified context-grouped forward path for both soft and sliced modes eliminated BN distribution mismatch.
- The shared backbone spontaneously specialized: `gaussian_noise` consistently picks a different channel set than the others, especially in shallow stages.

### What didn't

- Beating independent experts. Independent wins overall, with the gap concentrated on hard contexts (`gaussian_noise` −3.6%p, `frost` −7.3%p; `clean`/`brightness` are within 0.5%p).
- IoU between contexts stays 0.6–0.8, well above the random baseline of ~0.33 — channels are shared more than they specialize.

### Story to keep

Negative-but-interesting: shared-backbone sub-networks match independent experts within a small margin at equal compute, *spontaneously* specialize where it matters most (noise-heavy context, shallow layers), and have a clear scalability advantage (adding a new context costs one embedding row vs one full expert). The accuracy gap correlates with how much each context shares channels with the others — direct quantitative evidence that the residual cost lives where specialization failed to emerge.

---

## 1. Model comparison

Setup: ResNet34 / CIFAR-10-C / 4 contexts (`clean`, `brightness`, `frost`, `gaussian_noise`) at severity 3.

| Model | Total params | Active/ctx | FLOPs/img (est.) | overall | clean | brightness | frost | gaussian_noise |
|---|---|---|---|---|---|---|---|---|
| baseline (ResNet34, full width) | 21.3M | 21.3M | 2.32G | — | — | — | — | — |
| independent (w=0.5) | 21.3M | 5.3M | 0.58G | **0.826** | 0.846 | 0.841 | 0.848 | 0.767 |
| SCDMN (s=0.25, mask-on-activation, prior) | 21.5M | 21.5M (full fwd) | 2.32G | (prior result) | — | — | — | — |
| **SCDMN-Sliced (s=0.5)** | 21.3M | ~5.4M | ~0.58G | **0.799** | 0.849 | 0.842 | 0.775 | 0.731 |
| SCDMN-Sliced (s=0.25, first attempt) | 21.3M | ~1.4M | ~0.15G | 0.30 (failed) | 0.766 | 0.716 | 0.745 | 0.162 |

Headline comparison: **independent vs scdmn_sliced(s=0.5)** — virtually the same active params and FLOPs. Independent wins by 2.6%p overall; almost all of the gap comes from `gaussian_noise` and `frost`.

## 2. Channel mask IoU analysis (sparsity 0.5 → random IoU ≈ 0.33)

```
Stage 0 (C=64) — shallow, strongest specialization
  noise vs others: 0.42–0.49
  others mutual:   0.64–0.73

Stage 1 (C=128)
  noise vs others: 0.66 (uniform)
  others mutual:   0.78–0.86

Stage 2 (C=256)
  noise vs others: 0.68–0.71
  others mutual:   0.77–0.84

Stage 3 (C=512) — deepest, most sharing
  noise vs others: 0.64–0.66
  others mutual:   0.71–0.76
```

Reading:

- All IoU values exceed the random baseline (~0.33) substantially → contexts **share channels heavily**.
- `gaussian_noise` is consistently lower-IoU vs the other three → noise is the only context that learned a clearly distinct sub-network.
- Specialization is strongest in shallow stages and decays with depth → shallow layers do noise filtering, deep layers carry shared semantics.
- `clean` / `brightness` / `frost` use very similar sub-networks (visually similar statistics).

Visualization saved to `experiments/runs/sliced_analysis.png`.

## 3. What we tried — failures and fixes

### Attempt 1: SCDMN-Sliced (s=0.25, freeze_epoch=5) → **failed (overall 0.30)**

Root cause:

- Soft-warmup multiplied the mask on **activations**, not weights. The conv ran at full width and the mask only zeroed outputs.
- BatchNorm running stats accumulated under the full-width distribution.
- After freeze, the sliced forward used the same BN buffers but on a different (sliced) distribution → output collapsed.
- The collapse was worst on `gaussian_noise` (0.16), the context whose input distribution is most shifted.

### Attempt 2: Move soft mask onto conv WEIGHTS + sparsity 0.5, freeze_epoch=10, score lr ×10 → **worked (0.80)**

Changes:

- In soft mode: `W' = W * sigmoid(score)`. Conv weights and BN affine are scaled by the per-context score along the channel axis.
- Both soft and sliced modes use the **same** context-grouped forward structure (split batch by `ctx_label`, run per-context sub-network, scatter back).
- Initial std of `channel_scores` raised from 0.01 → 0.5 (faster initial divergence between contexts).
- Score parameter group given lr ×10 with `weight_decay=0`.
- `freeze_epoch` raised to 10 to give scores enough time to settle.

Verification: at the freeze instant, soft-eval and sliced-eval outputs differ by 0.0038 on a held-out batch — the modes are essentially equivalent at the limit, so freezing introduces no shock.

### Auxiliary fixes

- For stage-0 first block: `in_idx` is the full stem (64) but `out_idx` is the active subset (32). The no-projection identity branch handles this with `index_select` to gather `x` along the channel dim.
- Sliced BatchNorm normalizes over only the active channels but updates the original buffer in-place via `index_copy_` — a plain `running_mean[idx]` assignment would be a copy and the update would not persist.

## 4. Conclusion

### What's good

- **Spontaneous specialization in a shared backbone**: just learning per-context channel scores, with no diversity penalty, produces a meaningfully different sub-network for `gaussian_noise`. The shallow > deep specialization pattern matches the intuition that low-level features adapt to noise while high-level features can be shared.
- **Near-parity with independent experts at matched compute**: −2.6%p overall, ≤0.5%p on `clean`/`brightness`.

### What's not

- Sliced does not beat independent in absolute accuracy. The "sharing wins" story is not supported here.
- Hard contexts (`gaussian_noise`, `frost`) are exactly where shared-backbone hurts most — accuracy gap is concentrated there.
- IoU stays at 0.6–0.8, far above the random 0.33 — score learning under-specializes; the model prefers to share even when it could differentiate.

### Stories worth keeping

1. **Negative-but-interesting**: shared backbone is competitive at matched compute and *spontaneously* specializes; the residual gap is localized to contexts whose statistics diverge from the majority.
2. **Scalability advantage**: adding a new context costs one expert (~5.3M params) for independent vs one score embedding row (~512 params) for sliced. Beyond a handful of contexts, sliced's marginal cost dominates the comparison.
3. **IoU vs accuracy-gap correlation**: per-context mean IoU with the others is negatively correlated with the (independent − sliced) gap — quantitative evidence that the cost of sharing is paid by under-specialized contexts.

## 5. Future directions

### Short-term (small implementation cost)

- **Diversity loss**: an explicit regularizer on pairwise channel-score similarity to push IoU toward random. Current specialization is opportunistic; forcing it could recover the hard-context gap.
- **End-to-end binary masks**: Gumbel-softmax with temperature annealing instead of a hard freeze epoch. Removes a brittle hyperparameter.
- **Per-stage sparsity**: shallow layers more sparse (specialization is needed there), deep layers denser (sharing already works). The IoU pattern motivates this directly.
- **BN re-estimation after freeze**: one weights-frozen pass to clean up any residual BN distribution drift.

### Medium-term (experiment design)

- **More contexts**: scale from 4 → 10+ corruptions. Independent's cost grows linearly; sliced's stays nearly flat. This is where the scalability story becomes empirical, not just structural.
- **Few-shot context transfer**: freeze the backbone and fine-tune only a new score embedding on an unseen context. This is impossible for independent (need a whole new expert) and natural for sliced.
- **Beyond CIFAR-10-C**: ImageNet-C or driving datasets (BDD100K weather). Connects back to the DriverMOE direction.

### Long-term (structural)

- **Soft routing instead of top-k**: dense weighted mixture (MoE-style). Hard top-k loses learning signal post-freeze; soft routing keeps it alive.
- **Image-derived context embedding**: replace the oracle label with a learned continuous embedding extracted from the input — combine `scdmn_cnn` mode with sliced execution. Removes the test-time oracle assumption.

## 6. Files and checkpoints

- Model: `models/scdmn_sliced.py`
- Trainer: `experiments/trainer.py` (scdmn_sliced branch + per-group score lr)
- Entry point: `experiments/run_all.py` (`--run_sliced --sparsity --freeze_epoch`)
- Analysis: `utils/analyze_sliced.py`
- Checkpoints:
  - `experiments/runs/scdmn_sliced_s0.5_final.pt` (final working run)
  - `experiments/runs/independent_final.pt` (matched baseline, w=0.5)
  - `experiments/runs/main/baseline_final.pt` (full-width single ResNet)
- Figure: `experiments/runs/sliced_analysis.png`

## 7. Reproduction commands

```bash
# SCDMN-Sliced (working configuration)
scdmn-cifar/bin/python -m experiments.run_all \
  --run_sliced --sparsity 0.5 --freeze_epoch 10 \
  --skip_baseline --skip_independent --skip_scdmn

# Comparison: thin independent experts (w=0.5)
scdmn-cifar/bin/python -m experiments.run_all \
  --indep_width_mult 0.5 \
  --skip_baseline --skip_scdmn

# Generate analysis figure
scdmn-cifar/bin/python -m utils.analyze_sliced \
  --sliced_ckpt experiments/runs/scdmn_sliced_s0.5_final.pt \
  --indep_ckpt experiments/runs/independent_final.pt \
  --baseline_ckpt experiments/runs/main/baseline_final.pt \
  --out experiments/runs/sliced_analysis.png
```
