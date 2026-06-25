# CUDA training performance investigation (Numerai, RTX 5090)

A record of an attempt to speed up CUDA LightGBM training **without changing the output**, on the large
Numerai dataset. The headline result is a **negative** one — no output-preserving speedup exists for this
workload — plus one small quality-preserving win and some reusable methodology lessons. Documented here so
the dead ends and the reasoning behind them aren't re-explored from scratch.

- **Hardware/build:** RTX 5090 (sm_120, 32 GB), CUDA 13.2, `gpu_use_dp=true`, `max_bin=255`.
- **Data:** Numerai `train.parquet` (2.75M rows × 2748 int8 features). Benchmarked on a 1.2M×1200 subset
  (dense kernel) and a 1M×2748 full-feature subset (sparse kernel — see the scale lesson).
- **Method:** interleaved / paired A/B (baseline vs change in alternating blocks, fresh processes), with
  mean/median/stdev and a paired or Welch *t*. GPU clocks float 247↔3090 MHz and can't be locked (no perms),
  so **only the interleaved A/B ratio is trustworthy** — never compare against a stale absolute time.

---

## TL;DR

1. **No output-preserving (bit-identical) speedup exists.** Five experiments on the dominant histogram
   kernel, both code paths, both scales — all ≤ 0%. The kernel is already well-optimized; at the real
   feature count the cost is the inherent *volume* of histogram work over 2748 features, not a tunable
   inefficiency.
2. **One small real win (quality-preserving, config-only):** `gpu_use_dp=False` → **+2.49%** at the full
   2748 features with **identical Numerai quality**. Float atomics are cheaper than double on the
   bottleneck kernel. Changes exact predictions (precision) but not model quality.
3. **For determinism:** `use_quantized_grad=true` is bit-reproducible run-to-run and statistically
   quality-equivalent on the Numerai metric, but only ~+1% faster at full scale.
4. **Biggest methodology lesson:** the dominant kernel and the speedup numbers change completely between a
   feature *subset* and the real feature count. Always profile at the real scale.

---

## The bottleneck

`nsys` kernel breakdown: histogram construction dominates (~81–82% of GPU kernel time). The synchronization
overhead that used to dominate is now ~7% (addressed by prior event-ordering work, PRs #25/#27). `ncu`:
SM throughput ~77–78%, L1 ~67–69%, **DRAM only 10–17%** → the kernel is **compute / shared-atomic bound**,
not DRAM-bandwidth bound. `FindBestSplitsForLeafKernel` has *no* atomics and is ~9–13%.

The histogram kernel accumulates per-bin gradient/hessian via `atomicAdd_block` (shared) then
`atomicAdd_system` (global). The cost is the atomic traffic + the data reads, and there is no way to reduce
the atomics without changing the FP accumulation order (→ not bit-identical).

---

## Failed experiments (and why)

| # | Experiment | Kernel / scale | Result | Why it failed |
|---|---|---|---|---|
| 1 | **Shared-mem carveout** (`cudaFuncAttributePreferredSharedMemoryCarveout` = max) to lift occupancy 2→3 blocks/SM | dense / 1200 feats | −1.15% (n.s.) | Kernel is **throughput-bound, not latency-bound** (SM ~78%). More occupancy doesn't help when execution units are saturated, and max-shared carveout *steals L1*, which the kernel uses heavily. |
| 2 | **`__ldg` / `__restrict__`** on the read-only gathers | dense / 1200 feats | −0.39% (paired *t*=−0.18) | Not load-bound; it's atomic-bound. An early +2.67% reading was pure **GPU-clock noise**, refuted by a higher-power paired re-run. |
| 3 | **Reduce `NUM_THREADS_PER_BLOCK` 504→252** (cut per-feature atomic contention) | dense / 1200 feats | **SIGFPE crash** | Numerai's int8 features have ~5 bins, so many features pack per partition (`max_num_column_per_partition` ≈ 500). `block_dim_y = NUM_THREADS_PER_BLOCK / max_num_column_per_partition = 252/500 = 0` → divide-by-zero. The 504 default is deliberately ≥ that. (So `block_dim_y` is already ~1 for low-bin data → within-block contention is already minimal.) |
| 4 | **Increase `NUM_DATA_PER_THREAD` 400→4000** (fewer blocks → fewer cross-block `atomicAdd_system` merges) | dense / 1200 feats | −2.79% | Fewer blocks **underutilize the GPU** (parallelism loss > merge-traffic savings). 400 is already well-balanced. |
| 5 | **`__ldg` / `__restrict__`** on the *sparse* kernel | **sparse / 2748 feats (real config)** | −0.90% (paired *t*=−1.11) | Correct kernel + correct scale this time, but the sparse kernel's L1 pressure isn't from the global reads `__ldg` targets. Still atomic/compute-bound. |
| — | **`FindBestSplits` block size** (256 threads for ~5-bin features looks wasteful) | — | **not tunable** | `NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER=256` is coupled to a fixed-size `BitonicArgSortDevice<…256,11>` for categorical-split sorting and sized shared buffers; lowering it caps categorical-sort capacity. Not a safe/generalizable knob. |

All output diffs for the non-crashing experiments were within the baseline's own run-to-run ULP envelope
(corr 1.0) — i.e. the changes were genuinely output-preserving; they just weren't faster.

---

## Key insights

- **The histogram kernel is irreducibly atomic/compute-bound.** Occupancy (carveout, block config) and load
  efficiency (`__ldg`) are the only bit-identical levers, and both give ~0% because neither addresses the
  actual bottleneck. Reducing the atomics requires changing accumulation order (not bit-identical) or
  integer packing (`use_quantized_grad`).

- **Scale + code-path change the answer completely (most important lesson).** On the 1200-feature subset the
  histogram uses `CUDAConstructHistogram**Dense**Kernel` (uint8 bins). At the real 2748 features, LightGBM
  bundles the low-cardinality features into a **sparse multi-value-bin** layout and runs
  `CUDAConstructHistogram**Sparse**Kernel` (uint16 bins) — a different kernel. *Every dense-kernel experiment
  above is irrelevant to the real use case.* And `use_quantized_grad` measured **+8.65% on the 1200-feature
  subset but only +0.93% at 2748 features**: at full scale the cost is the sheer volume of histogram work,
  not per-atomic cost, so the "fewer atomics" trick stops mattering. **Always profile at the real feature
  count first.**

- **Quantized's "corr 0.75 vs double" is not a quality loss.** That number is model-vs-model prediction
  similarity. On the actual Numerai metric (per-era corr with target), a paired test over 174 validation
  eras found double vs quant16/quant32 *statistically equivalent* (bootstrap 95% CI spans 0). Quantized is a
  *different but equally good* model.

- **Single precision is the cleanest real win.** `gpu_use_dp=False` → +2.49% at full features with identical
  Numerai quality (mean per-era corr 0.01328 for both). Cheaper float atomics on the bottleneck kernel.

- **CUDA double-path training is intermittently non-deterministic at scale.** FP atomic ordering perturbs
  per-bin sums by ~1 ULP, which rarely tips a near-exact split-gain tie and diverges the tree. It's
  bit-identical at small scale and only surfaces at ~1M+ rows; `deterministic=true` is documented CPU-only
  and isn't wired into the CUDA histogram. The int-atomic (`use_quantized_grad`) path is fully deterministic.

- **Benchmarking hygiene that mattered here:** GPU boost state alone moves wall-time ~2.7×, so absolute times
  are meaningless — only interleaved/paired A/B ratios are. A promising +2.67% turned out to be clock noise
  under a higher-power re-run.

---

## Recommendations

- **Speed:** set `gpu_use_dp=False` for ~+2–3% at identical quality (validate the score on your full
  pipeline first). No code change.
- **Reproducibility:** set `use_quantized_grad=true` (bins 16–32) for deterministic, quality-equivalent runs.
- **A larger speedup** would require a fundamental redesign of the sparse-histogram kernel (e.g. fewer
  atomics / a different reduction), which changes output within the library's existing non-determinism
  envelope — a substantial, reviewed effort, not a config tweak. Note: a warp-aggregation approach does *not*
  fit this kernel's layout (warps span features, not data rows, so there are no intra-warp bin collisions to
  combine).
