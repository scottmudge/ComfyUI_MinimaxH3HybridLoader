# MiniMax H3 `ref2va` vs `fl2va` — Safetensor Tensor Comparison Analysis

Two variants of the Minimax H3 audio+video DiT were compared tensor-by-tensor:
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (multimodal reference support, ~19.5 GB)
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (first/last-frame keyframe only; higher quality, per Minimax)

Both files share an identical key layout (932 tensors each), identical tensor shapes, and identical quantization (`int8` + `.comfy_quant` metadata, identical bit-for-bit) . The comparison was performed by streaming tensors one at a time via `safetensors.safe_open` (mmap-backed, so peak RAM ≈ 2 × the largest single tensor rather than 2 × the entire 19.5 GB model). Statistics computed per tensor: mean / max abs difference, **relative mean abs difference vs `fl2va`** (mean|Δ| / mean|fl2va|), RMS, and **flattened cosine similarity**.

## Headline result

The divergence between the two checkpoints is **not** an early-vs-late gradient along the 50 DiT blocks. It is overwhelmingly concentrated in **a single sub-component per block**: the **`adaln_proj.linear.*`** weights (the per-token / per-modality AdaLN modulation projections in `DiTBlock`). Every other block sub-component is functionally identical.

## Per-category aggregates (sorted by lowest minimum cosine = most-dissimilar first)

| category                       |   # | params       | relMean | minCos  | avgCos  | maxAbs |
|--------------------------------|----:|-------------:|--------:|--------:|--------:|--------:|
| **`final_layer.adaln_proj`**   |   1 |   86 016     | 1.610   | −0.830  |  0.390  |  9.105 |
| **`blocks.adaln_proj`**        | 100 | 43 545 600   | 0.753   | −0.812  |  0.126  | 55.59  |
| `final_layer.audio_out`        |   2 |    172 064   | 0.158   |  0.992  |  0.994  |  0.100 |
| `blocks.attn` (qkv/out/norms)   | 300 | 7 708 390 400 | 0.0120  |  0.997  |  0.9998 |  138   |
| `final_layer.video_out`        |   2 |    516 192   | 0.0471  |  0.999  |  0.999  |  0.095 |
| `blocks.mlp` (fc1/fc2)         | 200 | 11 562 252 800 | 0.0180 |  0.999  |  0.9997 |  136   |
| `audio_patch_proj`             |   2 |    177 408   | 0.0345  |  0.999  |  0.9995 |  0.055 |
| `token_refiner.blocks`         |  16 |    770 725 376 | 0.0164 |  0.999  |  0.9997 |  0.088 |
| `condition_proj`               |   2 |    27 530 496 | 0.0251  |  0.9998 |  0.9998 |  0.031 |
| `video_patch_proj`             |   2 |    521 472   | 0.0217  |  0.9998 |  0.9998 |  0.059 |
| `adaln_t_table`                |   1 |      8 200   | 0.0290  |  0.9998 |  0.9998 |  0.014 |
| `blocks.norm` (norm1/norm2)    | 100 |    537 600   | 0.00383 |  1.000  |  1.000  |  0.039 |
| `token_refiner.other`          |   1 |      5 376   | 0.00166 |  1.000  |  1.000  |  0.023 |
| `rope.inv_freq`                |   1 |         16   | 0       |  1.000  |  1.000  |  0     |

(All `.comfy_quant` byte tensors — 200 of them — are byte-identical between files; they encode the shared int8 quantization format.)

## What is identical (load either source; identical output)

- `rope.inv_freq` — bit-for-bit identical.
- All `.comfy_quant` tensors (200 of them) — bit-for-bit identical (the int8 format is shared).
- Token-refiner norms and `final_norm` — cosine 1.000.
- `blocks.{0..49}.norm1.weight`, `blocks.{0..49}.norm2.weight` — cosine 1.000 (relMean ≈ 0.004, just `bf16` rounding).
- `blocks.{0..49}.attn.q_norm.weight`, `.k_norm.weight` — cosine 1.000.

## What is *near*-identical (cos ≥ 0.9997): the bulk of the model

These categories together account for **≈ 97% of all model parameters** and are functionally indistinguishable:

- `blocks.{0..49}.attn.qkv_proj.weight` — avg cosine 0.9999, relMean ≤ 0.020 (the largest is block 49 at 0.0206; most are ≈ 0.018).
- `blocks.{0..49}.attn.out_proj.weight` — avg cosine 0.9997, relMean 0.008–0.020.
- `blocks.{0..49}.mlp.fc1.weight`, `.fc2.weight` — avg cosine 0.9997, relMean 0.013–0.021.
- `token_refiner.blocks.{0,1}.*` — all ≥ 0.9994. Importantly, `token_refiner` (which refines both text states **and** reference items via `condition_proj`) is essentially the same between the two checkpoints. This means **the reference pathway is structurally present in both models**; what differs is how the modality-conditioning modulates the residual stream.
- `condition_proj`, `video_patch_proj`, `audio_patch_proj`, `adaln_t_table` — cosine 0.999–1.000.

## What diverges dramatically: `adaln_proj.linear.*`

This is the smoking gun. The DiT-block AdaLN projection (`AdalnProj` in `comfy/ldm/minimax/model.py:185`) takes the time embedding and produces, in one linear layer, `expand × 6 × 3 modalities` channels of `(shift, scale, gate)` — i.e. it produces the per-token, per-modality modulation that mixes reference/text/audio/video tokens into the residual stream. This is precisely the piece that had to be retrained to make `ref2va` attend to reference tokens.

For `blocks.N.adaln_proj.linear.weight` (and `.bias`):

| metric                  | range across 50 blocks                    |
|-------------------------|--------------------------------------------|
| cosine similarity       | −0.74 to −0.81 (i.e. *negatively aligned*) |
| relative mean diff      | 0.73 to 0.77 (diff is ~75% of weight magnitude) |
| max abs difference      | up to **55.59** (block 32)                 |

These weights are essentially **uncorrelated** between the two checkpoints. The training clearly rewrote them wholesale to teach the model the new reference-modality mixture.

`final_layer.adaln_proj.linear.weight` is the single most-dissimilar tensor in the entire model (cosine −0.83, relMean 1.61). This is consistent with `ref2va`'s reported poor quality: the final adaln projection is where the per-stream (video / audio) output modulation is decided, and the broken training left this layer in a noticeably degenerate state.

## Output-head divergence (moderate, structured)

`fl2va` and `ref2va` also differ measurably in the final output heads — small in absolute terms but the largest deviation outside the `adaln_proj`:

- `final_layer.audio_out.weight`: relMean 0.199, cosine 0.997 — the audio output head drifted the most of the four output tensors.
- `final_layer.audio_out.bias`:  relMean 0.117, cosine 0.992.
- `final_layer.video_out.weight`: relMean 0.072, cosine 0.999.
- `final_layer.video_out.bias`:  relMean 0.022, cosine 1.000.

This is expected: training on reference conditioning shifts where the final output distribution lands. These are still highly correlated (≥ 0.992) so they are the *least* interchangeable dimension and an obvious candidate to leave as `fl2va`.

## Per-block trends (block indices 0..49)

The DiT-block `adaln_proj` is uniformly different across all blocks (no early-vs-late pattern): its relMean sits in a tight 0.73–0.77 band and cosine in −0.70 to −0.81 across the entire stack. There is a *very* mild trend — middle/late blocks (28..35) reach the lowest cosine (≈ −0.80) and the first two blocks (0, 1) are the "least altered" (cosine ≈ −0.79 / −0.75) — but it is a small effect (≤ 5%) and not a useful split boundary.

The `norm1` / `norm2` RMSNorm scales drift up monotonically in relMean from ~0.003 (block 0) to ~0.0049 (block 41) — still cosine 1.000 everywhere — so this is just a sub-1% scale drift, not a structural change.

## What does this mean for the hybrid strategy?

1. The bulk of the model's *quality-determining* weights (attention QKV/O, MLPs, RMSNorms, patch projections, rope) is **shared between the two checkpoints** to ≥ 0.9997 cosine. Swapping them is a no-op at the precision we care about.

2. Therefore the *quality* of `fl2va` and the *reference-handling capability* of `ref2va` only meaningfully diverge in three groups:
   - **`blocks.{0..49}.adaln_proj.linear.{weight,bias}`** — the only DiT-block group that differs meaningfully.
   - **`final_layer.adaln_proj.linear.{weight,bias}`** — the worst-discordant tensor; *and*
   - **`final_layer.{audio_out,video_out}.{weight,bias}`** — the output heads, only moderately different.

3. The `token_refiner` and `condition_proj` (which actually process reference tokens before they enter the stack) are essentially identical between the two checkpoints. If `ref2va` "knows" references, it does so via the AdaLN modulators — not via a separate reference-encoder.

### Suggested hybrid configurations (in order of "least invasive")

There is no guarantee any of these will actually incorporate references effectively without `ref2va`'s full AdaLN chain — only empirical testing within ComfyUI can confirm that. The node implemented alongside this report exposes every group individually so you can try them.

- **`fl2va_base` (default, no overlay)**: load `fl2va` outright. Same behaviour as the stock `Load Diffusion Model` node. Use as the quality baseline / control.
- **`fl2va + blocks.adaln from ref2va`**: keep all of `fl2va`, overlay only `blocks.{0..49}.adaln_proj.linear.*` from `ref2va`. This is the minimum overlay that *might* let references influence modality routing while keeping the FL2VA quality of attention, MLPs, and output heads intact. *Most promising single configuration.*
- **`fl2va + all adaln from ref2va`**: also overlay `final_layer.adaln_proj` from `ref2va`. The degenerate `final_layer.adaln_proj` is the *most* different tensor in the model — taking it from `ref2va` is likely to degrade quality more than taking block-adalns does. Treat this as a "max-reference, accept-quality-loss" knob, not the recommended default.
- **`fl2va + adaln + heads from ref2va`**: also swap output heads. Strictly worse for output-quality unless the heads are coupled to the `adaln_proj` you swapped in (i.e. you load *neither* together nor alone for partial swaps).

### What we *don't* know without running the model

- Whether `fl2va`'s AdaLN modulators, when faced with reference tokens they were not trained on, produce sensible modality routing at all (i.e. whether hybrid-A is even coherent). The fact that the user already reports `ref2va` → `fl2va` swap "just works" with references present, however, suggests that *fl2va's* AdaLN chain handles the extra tokens without blowing up — which in turn suggests overlaying *`ref2va`'s adaln only* on top of `fl2va` is the configuration most likely to combine both benefits.
- Whether a block-range hybrid (e.g. blocks 0–N from `ref2va`, N+1..49 from `fl2va`, or vice versa) is useful. Per-block analysis says **no**: there is no early-vs-late gradient — every block's adaln is uniformly rewritten, and every block's other weights are uniformly the same. A per-block split is unlikely to outperform a global "all blocks" overlay. Still, the node supports per-range overrides for completeness.

## Files written alongside this analysis

- `/tmp/opencode/minimax_h3_comparison.json` — full per-key diff statistics (kept in `~/tmp` for the session; re-runnable any time from `compare_minimax_h3.py`).
- `/tmp/opencode/minimax_h3_summary.txt` — top-N most similar / different tensors, exact-equal tensor list, category aggregates.
- `/tmp/opencode/minimax_h3_blockbreakdown.txt` — per-block, per-subcomponent relMean + cosine table.
- `/tmp/opencode/compare_minimax_h3.py` — the reusable comparison script.
- `/tmp/opencode/blockbreakdown.py` — the per-block breakdown script.
