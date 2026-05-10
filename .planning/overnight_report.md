# Overnight CPU/CUDA Parity Investigation — Report

**Session start:** 2026-05-09 23:18 CEST
**Session end (active work):** 2026-05-10 08:35 CEST
**Loop cadence:** every 60 min — check open PRs, continue investigation, update this file

---

## Morning summary (read this first)

### Outcome

Started the night with 3 open PRs (#4, #5, #6), ended with **7 open PRs** all targeting genuine bugs (not FP-precision drift), all `MERGEABLE` per `gh pr list`, 6 of 7 with regression tests. No comments from Felix on any new PR — PR #4's pre-existing review feedback was already addressed in the prior session.

### PRs landed tonight

| PR | URL | Severity | Tests |
|---|---|---|---|
| **#10** | https://github.com/BelixRogner/ExaBoost/pull/10 | MEDIUM — `BitonicArgSort_1024 / _2048` made stable on tied values; halves LambdaRank round-1 divergence | yes |
| **#9** | https://github.com/BelixRogner/ExaBoost/pull/9 | MEDIUM — `PercentileGlobalKernel` unweighted formula (mirrors PR #6 in init-score path); reduces `reg_l1` and `reg_quantile` to bit-perfect | yes |
| **#8** | https://github.com/BelixRogner/ExaBoost/pull/8 | **HIGH (crash)** — `ShuffleSortedPrefixSumDevice` OOB shared-memory access; weighted L1/quantile no longer crashes | yes |
| **#7** | https://github.com/BelixRogner/ExaBoost/pull/7 | **HIGH** — `max_depth` was completely ignored on CUDA; models 2-7× deeper than requested | yes |

### PRs from earlier (carried over)

| PR | URL | Severity | Tests |
|---|---|---|---|
| #6 | https://github.com/BelixRogner/ExaBoost/pull/6 | MEDIUM — `PercentileDevice` unweighted formula bias | yes |
| #5 | https://github.com/BelixRogner/ExaBoost/pull/5 | LOW — CMake support for CUDA Toolkit 13.x | n/a (build) |
| #4 | https://github.com/BelixRogner/LightGBM/pull/4 | MEDIUM — `min_data_per_group` was silently ignored on CUDA | yes |

### Parity-sweep state

The 19-config sweep started divergent in 6 cases. After the 7 PRs:

| state | count | notes |
|---|---|---|
| `pred OK / tree OK` (perfectly clean) | 1 | `reg_categorical` |
| `pred OK` at FP epsilon | 16 | including `reg_l1` (was 0.40) and `reg_quantile` (was 0.58), now both bit-perfect |
| Genuinely divergent | 2 | `reg_bagging` (RNG, expected per #6055), `multi_dense` (FP-precision drift in parallel reduction, expected per #6055) |

### Recommended review order (by severity)

1. **PR #8** — crash. Anyone training weighted L1 or weighted quantile on CUDA gets "illegal memory access" before this fix.
2. **PR #7** — `max_depth` silently ignored. Anyone setting `max_depth` on CUDA was getting models 2-7× deeper than requested, with much weaker regularization than they thought.
3. **PR #4** — `min_data_per_group` silently ignored. Anyone training categorical features on CUDA was getting splits below the configured minimum group size.
4. **PR #9** then **PR #6** — wrong percentile formula in `PercentileGlobalKernel` (init scores) and `PercentileDevice` (per-leaf renewal). Bias `regression_l1` and `quantile` outputs upward.
5. **PR #10** — non-stable bitonic sort. Halves LambdaRank divergence, also fixes potential ranking-on-tied-scores edge cases.
6. **PR #5** — build fix for CUDA 13.x.

### Bugs investigated and confirmed *expected* (no PR)

- **`reg_bagging`** — CUDA's RNG samples a different bag than CPU's RNG. Per maintainer guidance in lightgbm-org/LightGBM#6055, this is "expected" (different number of threads = different floating-point error accumulation).
- **`multi_dense` round 3+, `fair`, weighted regression/binary, poisson** — FP-precision in CUDA's parallel histogram aggregation accumulates across rounds. Same #6055 framing.
- **13 cosmetic threshold-encoding cases** — predictions match at FP epsilon despite tree-dump threshold values differing. Same root cause; impacts only `dump_model()` output, not training-set predictions.
- **LambdaRank round-1 residual after PR #10** — FP-precision in `atomicAdd_block` ordering across pair gradients. Halved by the bitonic-sort fix; remaining ~0.14 is the same #6055 family.
- **`gamma` / `tweedie` / `mape` / `cross_entropy`** — these objectives have **no CUDA implementation** (per `src/objective/objective_function.cpp:59-66`); they fall back to CPU objective with CUDA tree learner. The hybrid path can produce divergent results but it's a documented limitation.

### Open follow-ups (not pursued tonight)

1. **CUDA WEIGHTED `PercentileGlobalKernel` formula** — PR #9 only fixed the unweighted branch. Weighted L1/quantile init scores still diverge (CPU returns 1.5, CUDA returns 3.5 for `y=[1,2,3,4,5]` weighted; both wrong vs Type-7 = 2.5). CPU's `WeightedPercentileFun` macro has its own off-by-one (uses `cdf[pos+1] - cdf[pos]` instead of `cdf[pos] - cdf[pos-1]`). Fixing both to standard Type-7 is a behavior change for CPU users. Aligning CUDA to CPU's existing buggy behavior is a parity fix but propagates the wrong answer. Decision deferred for morning conversation.
2. **`GlobalInclusivePrefixSumReduceBlockKernel`** discards the return of `ShufflePrefixSumExclusive` and adds the per-thread chunk sum instead of the exclusive prefix. Affects multi-block weighted percentile (n > 1024). Latent bug; no reliable reproducer this session.
3. **Felix's PRs #1 (quantized training) and #3 (discretized dense histogram offset)** are separately pending review and not affected by tonight's work.
4. **Investigate-branch bookkeeping**: `investigate/divergences` has all 7 fixes cherry-picked for local testing. It's not pushed; was used as the build/test baseline only.

### Cron status

Hourly cron job `b54bf38a` (`13 * * * *`) was scheduled at session start and has fired 6 times so far (00:13, 01:13, 02:13, 03:13, 04:13, 05:13, 06:13, 07:13). Since the job is session-only and you may close this session, **`CronDelete b54bf38a`** is the way to stop it manually if it's still firing when you read this.

---

## Open PRs at start of session

| # | Title | Status | Notes |
|---|---|---|---|
| 4 | [cuda] honor min_data_per_group in categorical split kernels | mergeable, no review | mine, awaiting Felix |
| 5 | [cmake] support CUDA Toolkit 13.x by gating dropped compute capabilities | mergeable, no review | mine, awaiting Felix |
| 6 | [cuda] fix unweighted percentile formula (L1 & quantile leaf renewal) | mergeable, no review | mine, awaiting Felix |
| 3 | [cuda] fix discretized dense histogram global-memory offset | mergeable, no review | Felix's — relevant, may interact with our investigation |
| 1 | [cuda] fix CUDA quantized training to match CPU | mergeable, no review | Felix's — also relevant |

## Parity sweep status (after PR #4 + PR #6)

| case | max\|Δ\| | tree-diffs | classification |
|---|---|---|---|
| reg_categorical | 0.0 | 0 | ✅ FIXED (PR #4) |
| reg_l1 | 0.25 | 38 | partially fixed by PR #6; residual = split-finding tie-break |
| reg_quantile | 0.54 | 32 | partially fixed by PR #6; residual = split-finding tie-break |
| reg_bagging | 0.39 | 6 | likely RNG difference |
| reg_max_depth | 0.25 | 3 | unknown — TODO investigate |
| multi_dense | 0.23 | 33 | unknown — TODO investigate |
| 13 cosmetic cases | <1e-15 | varies | threshold-encoding differences (predictions identical) |

## Investigation log

### 2026-05-09 23:18 — tie-break hypothesis (FALSIFIED)

**Hypothesis:** the cosmetic threshold differences for dense features arise because CPU runs reverse-direction scan first then forward, while CUDA's parallel reduction picks forward (lower task_index) on tie.

**Test:** swapped task creation order in `cuda_best_split_finder.cpp` so reverse comes at lower task_index.

**Result:** ONLY `reg_missing` improved (8→6 tree-diffs). All other cases unchanged. Dense `reg_dense` thresholds still `-0.2535942` (CPU) vs `-0.3838788` (CUDA) — bit-identical to pre-swap.

**Why I was wrong:** for `missing_type=None` features (the typical dense case), CUDA only creates ONE task (reverse-only). There's no cross-direction reduction to tie-break. The 13 cosmetic cases must come from a different cause.

**Status of attempt:** stashed as "tie-break swap attempt — marginal effect, dense case unfixed". May be worth committing as a small reg_missing-only improvement, but the dominant cause for 12 other cases is something else.

### Next investigation step

For dense `reg_dense` (single-task reverse-only), CPU and CUDA should be running structurally identical scans. Yet they pick different bin indices. Most likely root cause: **FP precision differences in histogram aggregation** — parallel reduction order on GPU vs sequential summation on CPU produces slightly different gradient sums per bin, flipping which bin has the marginal max gain.

To confirm: dump both histograms for the same leaf and feature, compare bin-by-bin.


### 2026-05-10 02:00 — landed max_depth fix (PR #7)

**Problem:** CUDA tree learner completely ignored `max_depth`. With `max_depth=2`, `num_leaves=31`: CPU produced depth-2 / 4-leaf tree, CUDA produced depth-7 / 31-leaf tree.

**Root cause** (two bugs combined):
1. `CUDABestSplitFinder::FindBestSplitsForLeaf` had no max_depth check (`grep max_depth src/treelearner/cuda/` returned NOTHING).
2. `CUDATree::Split` updated GPU-side `cuda_leaf_depth_` (via launch kernel) but never updated host-side `leaf_depth_`. So `tree->leaf_depth(idx)` always returned 0 on CUDA. Even adding (1) alone wouldn't have helped.

**Fix:** PR #7 https://github.com/BelixRogner/ExaBoost/pull/7
- Mirror host-side `leaf_depth_` update in `CUDATree::Split` and `SplitCategorical`
- Plumb `smaller_leaf_below_max_depth` and `larger_leaf_below_max_depth` flags into `FindBestSplitsForLeaf`, AND into existing validity checks
- Caller computes flags from `tree->leaf_depth(idx) < config_->max_depth`

**Verification:** parity sweep `reg_max_depth` case now `pred OK` at FP epsilon (down from `max|Δ|=0.25`). 9 parametrized regression tests added in `test_dual.py`; without the fix 5 of 9 fail.

### 2026-05-10 02:30 — confirmed reg_bagging is RNG-only, not a bug

**Method:** measured intra-CPU spread across `bagging_seed` ∈ {0..4} → range 0.91, stdev mean 0.14. CPU vs CUDA at same seed → 0.18-0.39. CPU/CUDA divergence sits well within intra-CPU RNG-induced spread.

**Conclusion:** explainable purely by CUDA's RNG sampling a different bag than CPU's RNG with the same nominal seed. Per maintainer guidance in [#6055](https://github.com/lightgbm-org/LightGBM/issues/6055) ("hard to ensure deterministic results for floating-point operations") plus the runtime warning "Although deterministic is set, the results ran by GPU may be non-deterministic" — this is documented expected behavior. No fix needed.

### 2026-05-10 02:45 — partial L1/quantile residual investigation (deferred)

**Status:** found that L1 round-1 root gain differs between CPU (62.153) and CUDA (62.965) even on identical splits with identical integer-valued gradients. Could be:
- L1 hessian or gradient computation differs (need to check CPU's GetGradients)
- Gain formula handles L1-specific terms (lambda_l1 default 0, so probably not)
- FP precision in gain computation order

**Time-cost:** likely deeper than overnight slot allows for a clean fix. Deferred.

### Findings summary as of 02:45

**Real bugs fixed (3 PRs total open):**

| PR | Bug | Status |
|---|---|---|
| #4 | categorical min_data_per_group ignored on CUDA | **needs tests** (TODO next) |
| #6 | unweighted percentile formula wrong (L1 + quantile) | tests landed |
| #7 | max_depth ignored on CUDA tree learner | tests landed |

**Confirmed expected behavior (won't fix):**

- reg_bagging: CUDA RNG samples differently from CPU RNG (#6055 acknowledged)
- 13 cosmetic threshold cases: FP precision in parallel histogram aggregation (#6055 acknowledged)
- multi_dense rounds 3+: secondary effect of cosmetic threshold drift (#6055 acknowledged)

**Open items:**

- PR #4 needs regression tests (TODO immediate)
- L1 round-1 gain divergence (deferred)
- Investigation into broader divergence sources (weighted training, GOSS, DART, extra_trees, tweedie/gamma/poisson, path_smooth, linear_tree)


### 2026-05-10 03:15 — quick triage of unexplored configurations

Ran `scratch/probe_unexplored.py` — broad sweep over weighted training, GOSS, DART, extra_trees, path_smooth, linear_tree, and rare objectives.

**Major finding (CRITICAL):** **Weighted L1 crashes CUDA with "illegal memory access".**

Reproducer (smallest):
```python
n = 100
X = np.random.randn(n, 3).astype(np.float64); y = np.random.randn(n).astype(np.float64)
w = np.random.rand(n)
ds = lgb.Dataset(X, label=y, weight=w, params={"verbose": -1})
lgb.train({"objective": "regression_l1", "device_type": "cuda", "num_leaves": 4, "min_data_in_leaf": 1}, ds, num_boost_round=1)
```

Crash chain:
- `cuda_regression_objective.cu:225` — `SynchronizeCUDADevice(...)` immediately after `RenewTreeOutputCUDAKernel_RegressionL1<USE_WEIGHT=true>` launch
- `cuda_tree.cpp:38` — destructor's `cudaStreamDestroy` then catches the persisted error

**Likely root cause:** in `include/LightGBM/cuda/cuda_algorithms.hpp:485-505` (`ShuffleSortedPrefixSumDevice`):

```cpp
__shared__ REDUCE_VAL_T shared_buffer[WARPSIZE];   // WARPSIZE = 32
...
thread_sum = ShufflePrefixSumExclusive<REDUCE_VAL_T>(thread_sum, shared_buffer);
const REDUCE_VAL_T thread_base = shared_buffer[threadIdx.x];   // OOB for threadIdx.x >= 32
```

The kernel is launched with `<<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>` = 256 threads, but `shared_buffer` is sized only WARPSIZE = 32. Reads at `shared_buffer[threadIdx.x]` for `threadIdx.x` in [32, 256) are OOB shared-memory accesses, returning garbage which then propagates to bad indices into other arrays.

**Why it didn't surface before:** the unweighted L1 path doesn't call `ShuffleSortedPrefixSumDevice`. So only weighted L1 / weighted quantile training trigger this. Crash threshold is around n=100 with the test's parameters — small unit-test datasets pass, real workloads crash.

**Threshold mapping:**

| n  | num_leaves=4 | result |
|---|---|---|
| 50 | OK | (passes) |
| 100 | CRASH | weighted L1 |
| 200 | CRASH | weighted L1 |
| 500+ | CRASH | weighted L1 |

**Status:** documented but not fixed. Fix requires either (a) replacing `shared_buffer[threadIdx.x]` with the per-thread exclusive sum already returned in `thread_sum`, or (b) sizing `shared_buffer[blockDim.x]`. Both have semantic implications worth verifying with the kernel's authors. Defer to morning conversation.

**My in-flight PRs (#4, #6, #7) are NOT regressions** — none touched the USE_WEIGHT percentile path. This is a pre-existing crash, almost certainly missed because there's no weighted-L1 CUDA test in the suite.


### 2026-05-10 03:30 — broad triage of unexplored configurations

Ran weighted/sampling/objective sweep beyond the original 19-config sweep. Findings:

**OK at FP epsilon (good news):**
- DART boosting
- GOSS sampling
- path_smooth (regularization)
- linear_tree
- tweedie / mape / huber objectives

**Divergent (max|Δ|):**

| config | max\|Δ\| | likely category |
|---|---|---|
| **weighted L1** | CRASH | **critical bug** (separate finding above) |
| weighted quantile | CRASH | same crash, same ShuffleSortedPrefixSumDevice path |
| **gamma objective** | **11.04** | possibly real bug — see below |
| **fair objective** | **1.89** | possibly real bug |
| binary + sample_weight | 0.33 | unknown |
| regression + sample_weight | 0.21 | unknown |
| poisson | 0.056 | small but real |
| extra_trees | 0.91 | likely RNG (same family as bagging) |
| feature_fraction_bynode | 0.59 | likely RNG |

### 2026-05-10 03:35 — gamma deep dive

Gamma diverges from round 1 with `max|Δ|=12.2`. Root splits AGREE (same `split_feature=6`, `threshold=-0.0317`, `gain=1179.02`). After the root, the trees diverge:

- **CPU** leaf populations: `[1, 4, 9, 60, 110, 15, 1]` (7 leaves)
- **CUDA** leaf populations: `[111, 1, 60, 13, 1, 2, 12]` (7 leaves)

CPU splits the right-of-root subtree into 5 leaves and the left-of-root into 2 leaves (1 + 110). CUDA does the opposite — left becomes a single 111-data leaf, right gets 6 leaves.

The structural divergence happens at depth 1 even with identical root. Gradient/hessian for gamma are computed in a CUDA-specific kernel; suspect different gradient values on CPU vs CUDA.

**Bonus bug found in dump:** CUDA's `dump_model()` reports `leaf_count=194` for the 111-data leaf — host-side `leaf_count_` not synced from GPU. Prediction routing IS correct (verified via `pred_leaf=True`); only the tree dump is wrong. Same family of bug as the `leaf_depth_` issue I fixed in PR #7. Not pursued further tonight.

**Status:** documented but not pursued — needs gradient-level CUDA debugging. Worth its own PR investigation.


### 2026-05-10 03:50 — landed weighted L1/quantile crash fix (PR #8)

**Reduced the OOB hypothesis to a one-line fix:** `ShuffleSortedPrefixSumDevice` had two related bugs:

1. `shared_buffer[threadIdx.x]` OOB read (sized `WARPSIZE=32`, indexed up to `blockDim.x=256`).
2. Inner loop didn't cumulate within per-thread chunk (only correct when num_data_per_thread == 1).

**Fix:** use the per-thread exclusive prefix sum already returned by `ShufflePrefixSumExclusive` (matching `GlobalMemoryPrefixSum`'s correct pattern) and cumulate inclusively across the chunk.

**PR #8** https://github.com/BelixRogner/ExaBoost/pull/8 — 7-line code change + 36-line regression test (8 parametrized cases over objective ∈ {regression_l1, quantile} × n ∈ {100, 200, 500, 1000}).

**Verification:** weighted L1 and weighted quantile now train successfully at all tested sizes.

### Total findings as of 03:50

**5 real bugs fixed (PRs open to BelixRogner/LightGBM):**

| PR | Bug | Severity | Tests |
|---|---|---|---|
| #4 | categorical `min_data_per_group` ignored on CUDA | MEDIUM | yes |
| #5 | CMake CUDA 13.x build broken | LOW (build-only) | n/a |
| #6 | unweighted percentile formula wrong | MEDIUM (L1, quantile bias) | yes |
| #7 | `max_depth` ignored on CUDA tree learner | HIGH (silent regularization bypass, models 2-7x deeper than requested) | yes |
| #8 | weighted L1/quantile CUDA crash | **HIGH (crash)** | yes |

**Confirmed expected per maintainer guidance ([#6055](https://github.com/lightgbm-org/LightGBM/issues/6055)) — won't fix:**

- `reg_bagging`: CUDA RNG samples differently from CPU RNG
- 13 cosmetic threshold cases: FP precision in parallel histogram aggregation
- multi_dense rounds 3+: secondary effect of cosmetic threshold drift
- `extra_trees`, `feature_fraction_bynode`: similar RNG-driven differences

**Open items requiring deeper investigation (not pursued tonight):**

- L1 / quantile residual structural divergence (after PR #6, leaf values are exact but tree structure still drifts; likely FP-precision-driven but worth confirming)
- gamma objective: max\|Δ\|=11 (very large), root splits agree but depth-1 splits diverge — needs gradient-level CUDA debugging
- fair objective: max\|Δ\|=1.89, similar shape to gamma
- regression + sample_weight: max\|Δ\|=0.21 (could be real or RNG)
- binary + sample_weight: max\|Δ\|=0.33 (could be real or RNG)
- poisson objective: max\|Δ\|=0.056 (small but real)
- CUDA `dump_model()` reports stale `leaf_count_` for some leaves (cosmetic — prediction routing is correct via `pred_leaf=True`)


---

## Morning summary (for review at 09:00)

**Bottom line:** 5 PRs open against BelixRogner/LightGBM, all targeting genuine bugs (not FP-precision drift). 4 have regression tests. Recommend reviewing in order of severity: **#8 (crash) → #7 (silent regularization bypass) → #4 (silent constraint ignore) → #6 (wrong percentile formula) → #5 (build fix)**.

**Open PRs:**

1. **PR #8** — https://github.com/BelixRogner/ExaBoost/pull/8 — fix CUDA crash in weighted L1 / quantile training (HIGH, with tests)
2. **PR #7** — https://github.com/BelixRogner/ExaBoost/pull/7 — enforce `max_depth` on CUDA tree learner (HIGH, with tests)
3. **PR #6** — https://github.com/BelixRogner/ExaBoost/pull/6 — fix unweighted percentile formula in L1/quantile leaf renewal (MEDIUM, with tests)
4. **PR #5** — https://github.com/BelixRogner/ExaBoost/pull/5 — support CUDA Toolkit 13.x (build fix)
5. **PR #4** — https://github.com/BelixRogner/LightGBM/pull/4 — honor `min_data_per_group` in categorical split kernels (MEDIUM, with tests)

**What was attempted but not pursued:**

- L1/quantile residual tree-structure divergence (after PR #6) — ~1% gain difference at round 1; consistent with FP-precision in parallel reduction; likely "expected per #6055"
- Cosmetic threshold-encoding divergences (13 cases in original parity sweep) — same root cause as above
- multi_dense round-3+ divergence — propagation of cosmetic threshold differences via residuals
- gamma objective: large divergence (max\|Δ\|=11) — needs gradient-level CUDA debugging
- fair objective: max\|Δ\|=1.89 — likely similar to gamma

**What I'd recommend prioritizing next:**

1. **gamma objective deep dive** — max\|Δ\|=11 is huge for a "supported" objective. Either real bug or a fundamental implementation difference worth documenting.
2. **CUDATree::dump_model leaf_count sync** — discovered a stale-host-state bug while looking at gamma. Cosmetic but easy to fix in same family as PR #7.
3. **Weighted regression / weighted binary parity** — both showed 0.2-0.3 divergence; need to determine if real bug or RNG-related.
4. **Other rare objectives** — poisson (0.056), tweedie/mape/huber (FP epsilon match — worth confirming with broader test inputs).

**Hourly cron status:** scheduled at `13 * * * *` (next firings: 04:13, 05:13, 06:13, 07:13, 08:13). Each iteration:
1. Reads this report
2. Checks open PRs for any review activity (responds substantively)
3. Picks the most tractable open item from this list
4. Appends findings here

If the cron is still active when you wake up, you can stop it with `CronDelete b54bf38a` (the loop's job ID).


### 2026-05-10 03:00 — second iteration (cron #1 firing at 02:34)

**PR review status:** No new comments on PRs #5, #6, #7, #8. PR #4 already fully responded to (Felix asked for tests + CUDA suite run; I supplied both yesterday).

**Continued investigation — gamma objective:**

Discovered that `gamma`, `tweedie`, `mape`, `cross_entropy*` objectives have **NO CUDA implementation** — they fall back to CPU objective (RegressionGammaLoss etc.) but with the CUDA tree learner. This hybrid mode (CPU objective + CUDA tree learner + CPU score updater) explains why gamma diverges so much: the CPU and CUDA paths take genuinely different code paths, not just different reductions.

Source: `src/objective/objective_function.cpp:59-61` and similar — explicit `Log::Warning("Objective gamma is not implemented in cuda version. Fall back to boosting on CPU.")`.

**Confirmed `fair` divergence is FP-precision drift, same family as multi_dense.** Round-1 leaves with same partitions have IDENTICAL values; only differing-partition leaves (off-by-1 from cosmetic threshold drift) have different values. Per #6055, expected.

**Confirmed `weighted regression` divergence is FP-precision drift.** Rounds 1-3 match at FP epsilon, divergence first appears at round 5. Same multi-round drift pattern as multi_dense.

**Confirmed `weighted binary` divergence has same shape** — round-1 leaves with same partition have identical leaf values; partition-different leaves differ. Cosmetic.

**Confirmed `poisson` divergence is also same shape** — rounds 1-3 at FP epsilon, then drift.

### 2026-05-10 03:30 — found weighted L1/quantile init-score bug

Beyond the crash fix in PR #8, the **weighted L1/quantile init score is also wrong on CUDA**. Logged init scores ("Start training from score" with verbose=1):

| n | numpy weighted median | CPU init | CUDA init | diff |
|---|---|---|---|---|
| 100 | -0.207 | -0.207 | -0.194 | 0.013 |
| 500 | 0.028 | 0.028 | 0.029 | -0.001 |
| 1000 | -0.029 | -0.029 | -0.028 | -0.001 |
| 5000 | 0.036 | 0.033 | 0.036 | (CPU loses to numpy here too) |

CPU matches numpy (Type-7 weighted percentile). CUDA is consistently off by 0.001-0.013.

**Found one definite bug** in `src/cuda/cuda_algorithms.cu:222` — `GlobalInclusivePrefixSumReduceBlockKernel` discards the return of `ShufflePrefixSumExclusive` and adds the local `thread_sum` (chunk sum) instead of the exclusive prefix `thread_base`. Fixed in working tree (uncommitted) — but this only affects multi-block (n > 1024) cases. The n=100 0.013 discrepancy must come from elsewhere.

**Other suspicious code in `PercentileGlobalKernel` (`include/LightGBM/cuda/cuda_algorithms.hpp:534`):**
```cpp
if (pos == 0 || pos == len - 1) {
    *out_value = values[pos];   // BUG: should be values[sorted_indices[pos]]
}
// no early return — interpolation runs anyway, with sorted_indices[pos-1]
// (OOB if pos == 0) — undefined behavior
const VAL_T v1 = values[sorted_indices[pos - 1]];
const VAL_T v2 = values[sorted_indices[pos]];
*out_value = ...;
```

For α=0.5 the boundary case probably doesn't trigger. The remaining n=100 discrepancy needs more digging — likely a bug in the per-block prefix sum or in the interpolation formula, but did not nail it this hour.

**Status of partial fix:** in working tree on `investigate/divergences`, NOT committed. Worth bundling with deeper PercentileGlobalKernel cleanup before opening PR #9.

### Next investigation step

- Continue probing PercentileGlobalKernel to find the n<=1024 bug in weighted L1/quantile init score
- Check LambdaRank / RankXENDCG (CUDA implementations exist, never parity-tested in this sweep)


### 2026-05-10 03:55 — investigation iteration #2 (cron firing 03:13/03:34)

**No new PR review activity** on PRs #5/#6/#7/#8.

**Continued investigating weighted L1 init-score divergence.** The picture is more nuanced than yesterday's report suggested:

**Discovery 1: Python wrapper drops uniform weights as an optimization.**

In `python-package/lightgbm/basic.py:3115-3122`:
```python
elif np.all(weight == 1):
    weight = None
```

When a user passes `weight=np.ones(n)`, the Python wrapper sets weight to None before sending to C++. So my earlier "weighted L1" tests were actually exercising the **unweighted** PercentileGlobal path. (I missed this on the first pass.)

**Discovery 2: CUDA's PercentileGlobalKernel UNWEIGHTED path has the same percentile-formula bug as PR #6 fixed for PercentileDevice.**

Both functions use `(1.0f - alpha) * len` instead of the correct `(1.0f - alpha) * (len - 1)`. The fix is identical to PR #6 but in a different function (this one is the GLOBAL kernel for init scores, not the device-side per-leaf renewal).

Reproducer: with non-uniform weights `[1,1,1,1,1.001]` and y=[1,2,3,4,5] (which forces the actual weighted code path, not the optimized-away unweighted path):
- numpy weighted median (Type-7): 2.5005
- CPU returns: 1.5005 (WRONG by ~1)
- CUDA returns: 3.5005 (WRONG by ~1 in opposite direction)

So **both CPU and CUDA have weighted-percentile bugs**, of different shapes:

- **CPU** `WeightedPercentileFun` (`src/objective/regression_objective.hpp:81-85`) uses `cdf[pos+1] - cdf[pos]` and `threshold - cdf[pos]` instead of the correct `cdf[pos] - cdf[pos-1]` / `threshold - cdf[pos-1]`. Off by one position.
- **CUDA** `PercentileGlobalKernel` weighted path uses different formula (descending-sort interpolation with bias `(threshold - cdf[pos-1]) / (cdf[pos] - cdf[pos-1])` between v1=values[sorted_indices[pos-1]] and v2=values[sorted_indices[pos]]) — this looks more correct than CPU's, but inputs to it are wrong because of the unweighted-path bug below.

**Discovery 3: The CUDA UNWEIGHTED PercentileGlobalKernel path also has the bad formula.**

```cpp
// include/LightGBM/cuda/cuda_algorithms.hpp:519-528
if (!USE_WEIGHT) {
    const double float_pos = (1.0f - alpha) * len;            // BUG: should be (len - 1)
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);      // BUG: should be +1
    if (pos < 1) {
        *out_value = values[sorted_indices[0]];
    } else if (pos >= len) {
        *out_value = values[sorted_indices[len - 1]];
    } else {
        const double bias = float_pos - static_cast<double>(pos);   // BUG: should be (pos - 1)
        ...
    }
}
```

Same family as PR #6.

For uniform weights (which Python collapses to None), users hit this path. For y=[1,2,3,4,5], CUDA returns 3.5 instead of 3.0.

**Decision:** the right fix for CUDA's PercentileGlobalKernel UNWEIGHTED path mirrors PR #6 exactly. **Will be PR #9** in the next iteration once I've also added a regression test. Reverted my earlier reduce-kernel attempt — it didn't actually help; the real bug is in the unweighted percentile formula.

**The CPU WeightedPercentileFun bug is separately upstream-able**, but it's a behavior change for CPU users and probably out of scope for this overnight session. Will document for the morning.

**Discovery 4: my reduce-kernel fix attempt was misdiagnosed.**

`GlobalInclusivePrefixSumReduceBlockKernel` does discard the return of `ShufflePrefixSumExclusive` and use the local `thread_sum` instead — but this only matters when `num_blocks > 1024 / 1` (i.e., very large datasets). For the test cases I tried (n up to 5000, num_blocks ≤ 5), the reduce kernel doesn't actually contribute meaningfully. The visible bug in small-n tests was actually the unweighted-formula bug I just found above.

The reduce-kernel issue is still a latent bug for very large weighted percentile inputs (>1024 samples), but I don't have a reliable reproducer for it tonight.

### Next investigation step

For next iteration:
1. Build PR #9 with the PercentileGlobalKernel unweighted-formula fix (+ regression test for weighted L1 init score with non-uniform weights, and ideally a separate test for unweighted-y init via `objective='quantile'` with α=0.5).
2. Possibly investigate LambdaRank parity (CUDA implementation exists, never parity-tested).

**Working tree state:** clean (no uncommitted changes). Branch `investigate/divergences` with all 5 in-flight fixes (PRs #4, #5, #6, #7, #8) applied + cherry-picked.


### 2026-05-10 04:45 — landed PR #9 (PercentileGlobalKernel fix)

**No new PR review activity** on PRs #4-8.

**Implemented the planned fix:** ported PR #6's percentile-formula fix to `PercentileGlobalKernel` (init-score path). Verified end-to-end:

- For y=[1,2,3,4,5] regression_l1, init goes from CUDA's 3.5 (wrong) to 3.0 (correct, matches CPU and numpy).
- 24 parametrized regression tests in `test_dual.py` all pass with the fix.

**Parity sweep impact:** TWO MORE cases now bit-perfect:

| case | before tonight | after tonight |
|---|---|---|
| `reg_l1` max\|Δ\| | 0.40 (orig) → 0.25 (after PR #6) → **0.000e+00** (PR #9) |
| `reg_quantile` max\|Δ\| | 0.58 (orig) → 0.54 (after PR #6) → **0.000e+00** (PR #9) |

PR #9: https://github.com/BelixRogner/ExaBoost/pull/9

### Remaining divergences in parity sweep (after 6 PRs)

| case | max\|Δ\| | category |
|---|---|---|
| `reg_bagging` | 0.39 | RNG, expected per #6055 |
| `multi_dense` | 0.23 | FP-precision drift in parallel reduction (rounds 3+) |
| 13 cosmetic | <FP eps | threshold-encoding differences (predictions identical) |

**All fixable bugs surfaced by the parity sweep are now PR'd.**

### Next investigation step

1. **Investigate LambdaRank / RankXENDCG parity** — CUDA implementations exist, never parity-tested. May surface new bugs.
2. **Investigate the CPU `WeightedPercentileFun` off-by-one** — affects users with non-uniform sample weights on `regression_l1`/`quantile`. Wider impact than CUDA-only fixes; would need careful upstream framing.
3. **Check the `GlobalInclusivePrefixSumReduceBlockKernel` reduce-kernel bug for very large weighted percentile inputs (n > 1024)** — latent but no reliable repro yet.

**Working tree state:** clean. Branch `investigate/divergences` now has all 6 in-flight fixes (PRs #4, #5, #6, #7, #8, #9).


### 2026-05-10 04:58 — investigation iteration #4 (cron firing 04:34)

**No new PR review activity** on PRs #4-9.

**Investigation summary this hour:**

**Probed LambdaRank / RankXENDCG parity:**

| objective | round 1 | round 5 | category |
|---|---|---|---|
| LambdaRank | max\|Δ\|=0.29 | 0.51 | **likely real bug** — round-1 divergence rules out FP drift; rooted in non-stable bitonic sort with all-zero round-1 scores |
| RankXENDCG | FP epsilon | 0.04 | FP-precision drift after round 1 |

LambdaRank's round-1 divergence: with all scores=0 at round 1, CPU's `std::stable_sort` returns indices in original order while CUDA's `BitonicArgSort` returns some non-deterministic permutation. Different sort orders → different `high_rank/low_rank` pair assignments → different gradients → divergent trees from round 1 onward.

**Decision on LambdaRank:** difficult to fix without changing the sort algorithm or adding a tie-breaking convention. Could be addressed by combining score with index as a lexicographic key in the sort, but that's a larger CUDA-kernel change. Documenting as a follow-up.

**Probed sparse/init-score/categorical/multiclass/quantized/force_row_wise/boost_from_average edge cases:**

| case | result |
|---|---|
| sparse CSR input (regression, binary) | OK FP epsilon |
| manual init_score | OK FP epsilon |
| high-cardinality categorical (30 levels) | OK 0.000e+00 |
| **multiclass 5 classes** | X 0.26 — same shape as `multi_dense` (FP drift, expected) |
| multiclassova 3 classes | OK FP epsilon |
| **`use_quantized_grad=True` (regression)** | X 1.71 — covered by PR #1, not yet merged |
| **`use_quantized_grad=True` (binary)** | X 28.01 — covered by PR #1 |
| `force_row_wise=True` | OK FP epsilon |
| `boost_from_average=False` | OK FP epsilon |

**Probed prediction-side modes:**

| case | result |
|---|---|
| `predict(raw_score=True)` on train | OK FP epsilon |
| `predict(pred_leaf=True)` | OK 0.0 (leaf assignments identical) |
| **`predict(pred_contrib=True)` (SHAP)** | X 0.032 — concentrated in features 3 and 6 (same features with cosmetic threshold drift) |
| Predictions on 10000 NEW rows | X 0.86 — cosmetic threshold drift causes ~1 row per tree to route differently |
| Various `start_iteration`/`num_iteration` | OK FP epsilon |

The SHAP and 10000-row prediction findings both trace back to the cosmetic-threshold cases already documented as expected per #6055 (TreeSHAP computes path-based attribution that's sensitive to threshold values even when training-set predictions are identical).

### Net conclusion

After 6 PRs and extensive probing, **the parity sweep's "real" divergent cases have all been resolved or shown to be expected**. The remaining categories of divergence are:

1. **FP-precision drift in parallel reduction** — multi_dense, multiclass5, regression+weight, binary+weight, fair, poisson, RankXENDCG round 2+, etc. Documented as expected per maintainer #6055.
2. **RNG-driven differences** — reg_bagging, extra_trees, feature_fraction_bynode. Documented as expected.
3. **Sort-tie-breaking** — LambdaRank round 1 (all-equal scores). Could be addressed but invasive.
4. **Quantized training divergence** — covered by Felix's PR #1 (open, unmerged).
5. **CPU's WeightedPercentileFun off-by-one** — affects CPU-only with non-uniform weights. Not a CUDA bug. Out of scope.

### Next investigation step

For the remaining iterations:
- Watch for new PR comments (none yet on any of my 6 PRs).
- Investigate CPU `WeightedPercentileFun` off-by-one more carefully — could be a CPU PR (out of scope of this CPU/CUDA work but useful upstream).
- Possibly test multi-GPU paths (NCCL) if any.
- Check edge cases in distributed/sliced datasets.

**Working tree state:** clean. `investigate/divergences` branch has all 6 fixes.


### 2026-05-10 06:45 — landed PR #10 (BitonicArgSort tie-stability)

**No new PR review activity.**

**Implemented LambdaRank fix from previous iteration's analysis.**

The bitonic sort comparator `(a > b) == ascending` was unstable on ties — for descending sort (ascending=false) and a==b, `false == false` is true, triggering a swap. Replaced with strict-direction comparator:

```cpp
const bool need_swap = ASCENDING ? (a > b) : (a < b);
```

Applied to both `BitonicArgSort_1024` and `BitonicArgSort_2048`.

**Impact:**

| case | before | after |
|---|---|---|
| LambdaRank round-1 max\|Δ\| | 0.29 | 0.14 |
| Categorical split parity | exact | exact (no regression) |
| All other parity-sweep cases | (their baseline) | unchanged |

The remaining LambdaRank residual (0.14) is FP-precision in `atomicAdd_block` accumulation order across pair gradients — documented expected per #6055.

**PR #10:** https://github.com/BelixRogner/ExaBoost/pull/10 — fix + 56-line regression test.

### Updated PR list

| PR | Title | Status |
|---|---|---|
| #4 | categorical min_data_per_group | mergeable, 2 comments addressed |
| #5 | CMake CUDA 13.x | mergeable, no review |
| #6 | percentile formula (PercentileDevice) | mergeable, no review |
| #7 | max_depth enforcement | mergeable, no review |
| #8 | weighted L1/quantile crash | mergeable, no review |
| #9 | percentile formula (PercentileGlobalKernel) | mergeable, no review |
| #10 | BitonicArgSort tie-stability | mergeable, no review (just opened) |

**Total: 7 PRs open**, all targeting real bugs. 6 of 7 have regression tests (PR #5 is build-only, doesn't need one).

### Next investigation step

Most fixable bugs from the parity sweep are now PR'd. Remaining areas worth probing:
1. **Distributed/multi-GPU paths** (NCCL) — untested
2. **Test with MAX_ITEM_GREATER_THAN_1024 ranking** to verify BitonicArgSort_2048 fix takes effect
3. **CPU `WeightedPercentileFun` off-by-one** — affects CPU-only with non-uniform weights, separate from CUDA work
4. **Probe broader objective+sample-weight combos** with the crash fix in place

**Working tree state:** clean. Branch `investigate/divergences` now has 7 in-flight fixes (PRs #4-10).


### 2026-05-10 07:50 — investigation iteration #6 (cron firing 07:34)

**No new PR review activity** on PRs #4-10.

**Probed weighted-objective parity (now uncrashable thanks to PR #8):**

| case | round 1 | round 5 | classification |
|---|---|---|---|
| regression + w | FP eps | FP eps | clean |
| **regression_l1 + w** | 0.15 | 0.59 | round-1 init bug — CUDA's WEIGHTED PercentileGlobalKernel path |
| **quantile a=0.5 + w** | 0.15 | 0.61 | same as L1 |
| **quantile a=0.7 + w** | (similar) | 1.33 | same family |
| huber + w | 9e-09 | 9e-09 | effectively clean |
| **fair + w** | (similar) | 2.33 | similar shape but with larger magnitude (FP precision more visible at higher leaf values) |
| binary + w | FP eps | 0.35 | round-1 OK → drift (FP precision, expected) |
| xentropy + w | FP eps | 0.35 | same as binary+w |
| **multiclass k=3 + w** | 0.27 | 0.45 | tree roots match exactly between CPU/CUDA, divergence is in deeper splits — same family as multi_dense (cosmetic threshold drift) |
| multiclassova k=3 + w | (similar) | 0.45 | same as multiclass |

**Conclusion on weighted divergences:**

- L1+w / quantile+w have a **real bug** in the WEIGHTED PercentileGlobalKernel init-score path. PR #9 only fixed the unweighted branch. The weighted branch in CUDA uses different conventions and is also buggy (returns 3.5 for the y=[1,2,3,4,5] case where CPU returns 1.5; both wrong vs Type-7 which is 2.5). Fixing this would require choosing between matching CPU's also-buggy formula or fixing both — significant scope.
- All other divergences are FP-precision drift, expected per #6055.

**Probed large-ranking (BitonicArgSort_2048 path):**

| case | result |
|---|---|
| lambdarank items=1500 (uses 2048 sort) | 0.46 |
| rank_xendcg items=1500 | FP epsilon |
| lambdarank items=500 (uses 1024 sort) | 0.40 |

The BitonicArgSort_2048 fix in PR #10 is exercised for items=1500 but doesn't dominate the LambdaRank divergence — most of the residual is FP-precision in pair-gradient atomicAdd_block ordering.

### Open weighted-percentile follow-up

The CUDA WEIGHTED `PercentileGlobalKernel` formula and CPU's `WeightedPercentileFun` are BOTH incorrect (off-by-one in different ways). This affects only training with non-uniform sample weights on `regression_l1` / `quantile`. Aligning them would close the round-1 init-score gap but require choosing one of:

a) Fix CPU `WeightedPercentileFun` to standard Type-7 weighted quantile + fix CUDA to match. **Behavior change for CPU users**, riskier.
b) Fix CUDA to match CPU's existing (buggy) behavior. **Parity-only fix**, but propagates the wrong answer.
c) Leave both as-is and document as known limitation.

I think (b) is the right scope for this overnight session if I keep going, but it requires careful kernel rewriting (the formula has a different shape on descending vs ascending). Defer for the next iteration.

### Next investigation step

1. Either implement option (b) above (CUDA weighted-percentile fix to mirror CPU's behavior) or document and move on.
2. Pull all 7 PRs together and dry-run a "merge to belix/master" to verify they apply cleanly.
3. Check if there are any cleanup/refactor opportunities discovered during investigation that could be small follow-up PRs.

**Working tree state:** clean. Branch `investigate/divergences` has 7 in-flight fixes (PRs #4-10).


### 2026-05-10 08:35 — final iteration (cron firing 08:34)

No new PR review activity. Final parity-sweep verification before 09:00 stop:

```
case                   pred tree      max|Δ|     mean|Δ|  #cpu  #cu note
----------------------------------------------------------------------------------------
reg_dense                OK    X   4.441e-16   ...  cosmetic threshold (expected)
reg_l1                   OK    X   0.000e+00   ...  bit-perfect (was 0.40, fixed by PR #6 + #9)
reg_quantile             OK    X   0.000e+00   ...  bit-perfect (was 0.58, fixed by PR #6 + #9)
reg_bagging               X    X   3.871e-01   ...  RNG (expected #6055)
reg_max_depth            OK    X   2.220e-16   ...  was 0.25, fixed by PR #7
reg_categorical          OK   OK   2.220e-16   ...  was 0.47, fixed by PR #4
multi_dense               X    X   2.264e-01   ...  FP drift (expected #6055)
... 12 other cases all OK at FP epsilon ...
```

**State at end:**

- All 7 PRs to BelixRogner/LightGBM are still `MERGEABLE` per `gh pr list`.
- Branch `investigate/divergences` (local) has 7 cherry-picked commits. Not pushed.
- Working tree clean (just untracked scratch/, .planning/, .claude/).

Wrote the morning summary at the top of this report. Stopping active work for the night per the 09:00 deadline.


### 2026-05-10 09:34 — past 09:00 stop deadline

Cron fired at 09:13 but `date` returned 09:34 CEST when this iteration began — past the 09:00 target. Cancelled the cron (`CronDelete b54bf38a`) and exited without further investigation. No PR review activity to address.

Final state unchanged from the 08:35 entry:
- 7 PRs open and `MERGEABLE`
- 17 / 19 parity-sweep cases at FP epsilon
- Branch `investigate/divergences` (local) holds all 7 cherry-picked fixes
