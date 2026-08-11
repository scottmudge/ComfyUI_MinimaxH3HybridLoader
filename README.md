# ComfyUI MiniMax H3 Hybrid Loader

A custom ComfyUI node that loads a MiniMax H3 (audio+video DiT) checkpoint by merging **selected tensor groups** from a second "overlay" checkpoint onto a "base" checkpoint, then hands the merged state dict to ComfyUI's stock `load_diffusion_model_state_dict`. The result is indistinguishable from a model loaded by `Load Diffusion Model` — same detection, same patcher, same multigpu deepclone support.

## Why

Minimax shipped two MiniMax H3 checkpoints with identical weight layout:

- **`minimax_h3_fl2va`** — trained on first/last-keyframe conditioning only; Minimax reports noticeably higher output quality.
- **`minimax_h3_ref2va`** — additionally trained on multimodal reference conditioning (image / video / audio references), but a confirmed training-quality issue makes its raw output worse.

A tensor-by-tensor comparison (see [`minimax_h3_analysis.md`](./minimax_h3_analysis.md)) shows **>97% of the weights** (attention QKV/O, MLPs, RMSNorms, patch projections, rope, token refiner) are bit-identical or cosine ≥ 0.9997 between the two. The only sub-component that differs meaningfully is the per-block **`adaln_proj.linear.*`** weights — the AdaLN modulation projections that route text / audio / video / *reference* modality tags through the residual stream — plus the `final_layer.adaln_proj.linear.*` weight (the single most discordant tensor in the model) and, to a lesser degree, the `video_out` / `audio_out` output heads.

So a promising hybrid is: load `fl2va` as the base (high-quality attention, MLPs, and output heads), and overlay *only* the per-block `adaln_proj` weights from `ref2va` (preserving the reference-conditioning pathway). This node exposes that configuration as its default preset, plus several coarser / finer presets and an explicit custom glob string so you can experiment.

## Recommended Settings for Higher Ref2VA Quality:

<img width="653" height="337" alt="good_settings" src="https://github.com/user-attachments/assets/cef3dbc1-0424-435f-99c9-4a5ed8d337ca" />

After testing this is what I, subjectively, think is the best in terms of reference capability and visual/audio quality:

* **fl2va** model as **base**
* **ref2va** model as **overlay**
* **block_range_adaln** overlay preset
* **block_range_start** set to **30**
* **block_range_end** set to **49**
  
All other settings default

## Features

- **Memory-friendly:** both safetensors files are opened mmap-backed and tensors are streamed one key at a time. Peak RSS is one model's worth (~19.5 GB for the int8 checkpoints), not 2× — the same as the stock loader.
- **Read-only:** neither safetensors file is mutated on disk.
- **Stock-compatible:** behaves exactly like `Load Diffusion Model` when `overlay_preset == "none"`.
- **Quantization-aware:** int8 `.comfy_quant` siblings always co-travel with the weight they belong to.

## Presets

| Preset | What it does |
|---|---|
| `none` | Pure base loading (equivalent to stock `UNETLoader`). |
| `ref2va_adaln_over_fl2va` *(default)* | Take per-block `adaln_proj.linear.*` from the overlay only. The recommended hybrid. |
| `ref2va_all_adaln_over_fl2va` | Also take `final_layer.adaln_proj` from the overlay. "Max-reference, accept-quality-loss" knob. |
| `ref2va_full_over_fl2va` / `fl2va_full_over_ref2va` | Take everything from the overlay (sanity checks). |
| `block_range_adaln` | Take `adaln_proj` only for blocks in `[block_range_start, block_range_end]` (inclusive, 0..49). |
| `custom` | Use `custom_overlays` / `custom_base` only. |

## Optional inputs

- **`block_range_start` / `block_range_end`** — Only used with `block_range_adaln`. The MiniMax H3 DiT has 50 blocks indexed 0..49.
- **`final_adaln_from_overlay`** — Additive toggle (independent of preset) to pull `final_layer.adaln_proj.linear.*` from the overlay on top of whatever the preset already does.
- **`custom_overlays`** — Comma-separated keys / prefixes / globs to *also* take from the overlay on top of the preset. Bare prefixes ending in `.` match by prefix (e.g. `blocks.49.`); other strings are matched as fnmatch globs (e.g. `blocks.[0-4].*.attn.qkv_proj.weight`).
- **`custom_base`** — Comma-separated keys / prefixes / globs to force *back* to the base even if the preset or `custom_overlays` would take them from the overlay.
- **`weight_dtype`** — Same meaning as the stock `Load Diffusion Model` node (`default`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`).

## Installation

Drop this repository into `ComfyUI/custom_nodes/` and restart ComfyUI. The node appears under the **model/loaders** category as **MiniMax H3 Hybrid Loader**.

## Further reading

The full per-tensor comparison of `ref2va` vs `fl2va`, including per-block trends and the rationale for each suggested hybrid configuration, is in [`minimax_h3_analysis.md`](./minimax_h3_analysis.md).
