"""MiniMax H3 hybrid diffusion-model loader.

This module provides a ComfyUI node that loads a MiniMax H3 (audio+video DiT)
safetensors checkpoint, but instead of reading every tensor from a single
checkpoint file it reads *selected* tensors from a second "overlay" checkpoint
and the rest from a primary "base" checkpoint. The merged state dict is then
handed to ComfyUI's stock ``load_diffusion_model_state_dict`` so the result is
indistinguishable from a model loaded by ``Load Diffusion Model`` -- same
detection, same patcher, same multigpu deepclone support.

Motivation
----------
Minimax shipped two MiniMax H3 checkpoints with identical weight layout:

  * ``minimax_h3_fl2va``  -- trained on first/last-keyframe conditioning only;
                              reports much higher output quality.
  * ``minimax_h3_ref2va`` -- additionally trained on multimodal reference
                              conditioning (image / video / audio references),
                              but Minimax confirmed a training-quality issue
                              that makes its raw output noticeably worse.

Tensor-by-tensor comparison (see ``minimax_h3_analysis.md``) shows that
>97% of the weights (attention QKV/O, MLPs, RMSNorms, patch projections,
rope, token refiner) are bit-identical or cosine->=0.9997 between the two.
The only sub-component that differs meaningfully is the per-block
``adaln_proj.linear.*`` weights (the AdaLN modulation projections that route
text / audio / video / *reference* modality tags through the residual
stream) and the ``final_layer.adaln_proj.linear.*`` weight (the most
discordant tensor in the model). The output heads ``video_out``/``audio_out``
differ only moderately.

So a promising hybrid is: load ``fl2va`` as the base (high-quality attention
+ MLPs + output heads), and overlay *only* the per-block ``adaln_proj``
weights from ``ref2va`` (preserving the reference-conditioning pathway).
This node exposes that and several coarser / finer presets, plus an explicit
custom prefix string, so the user can experiment.

Memory
------
With DynamicVRAM/AIMDO enabled, each source checkpoint uses one of ComfyUI's
managed ``ModelMMAP`` objects. Only the selected tensors are exposed to the
merged state dict, and ComfyUI can bounce file-backed pages between execution
stages just as it does for the stock diffusion-model loader. Without AIMDO the
loader falls back to read-only per-tensor mappings. Neither path copies the
selected full-width AdaLN tensors into ordinary RAM.

This node does not mutate either safetensors file on disk; both files are
opened read-only.
"""

from __future__ import annotations

import ctypes
import fnmatch
import json
import logging
import mmap
import os
import re
import struct
import threading
import warnings
from math import prod
from typing import Callable

import torch

import comfy.sd
import comfy.memory_management
import comfy.utils
import comfy_aimdo.model_mmap
import folder_paths


# ---------------------------------------------------------------------------
# Tensor-group spec
# ---------------------------------------------------------------------------

# Each "group" is a function from a state-dict key to a bool. Presets below
# turn the user-facing preset name into an ordered list of (group, source)
# pairs applied to the merged state dict (later overrides earlier; the "base"
# source is whatever the base checkpoint contributes, the "overlay" source is
# whatever the overlay checkpoint contributes). Matches are evaluated on the
# raw, fully-qualified state-dict key (e.g. ``blocks.3.adaln_proj.linear.weight``).

def _match_block_adaln(key: str) -> bool:
    """blocks.<N>.adaln_proj.linear.{weight,bias} for any N."""
    return bool(re.match(r"blocks\.\d+\.adaln_proj\.linear\.(weight|bias)$", key))


def _match_final_adaln(key: str) -> bool:
    return key in ("final_layer.adaln_proj.linear.weight",
                   "final_layer.adaln_proj.linear.bias")


def _match_final_heads(key: str) -> bool:
    return key.startswith("final_layer.video_out.") or key.startswith("final_layer.audio_out.")


def _match_all(key: str) -> bool:
    return True


def _match_blocks(key: str) -> bool:
    return key.startswith("blocks.")


def _match_token_refiner(key: str) -> bool:
    return key.startswith("token_refiner.")


def _match_input_projections(key: str) -> bool:
    return (key.startswith("video_patch_proj.") or
            key.startswith("audio_patch_proj.") or
            key.startswith("condition_proj."))


def _match_finalize(key: str) -> bool:
    return key.startswith("final_layer.")


def _make_block_range_adaln_matcher(start: int, end: int) -> Callable[[str], bool]:
    """Build a matcher that accepts ``blocks.<N>.adaln_proj.linear.{weight,bias}``
    for ``start <= N <= end`` (inclusive).

    The minimax h3 DiT has 50 blocks indexed 0..49. Passing a wider range
    is harmless -- the matcher simply never reports True for any index that
    doesn't exist in the state dict (so 0..49 is effectively the maximum
    useful window).
    """
    lo = max(0, int(start))
    hi = max(lo - 1, int(end))  # hi >= lo - 1 makes an inverted range just empty

    pattern = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$")

    def _m(key: str) -> bool:
        m = pattern.match(key)
        if not m:
            return False
        idx = int(m.group(1))
        return lo <= idx <= hi

    return _m


# Preset -> list of (matcher, overlay-or-not). The first preset is the
# identity (pure base loading, behaves exactly like the stock UNETLoader).
# "ref2va_adaln_over_fl2va" is the recommended hybrid per the analysis:
# take fl2va as the high-quality base, overlay only the per-block
# ``adaln_proj.linear.*`` weights from ref2va (preserving the
# reference-conditioning pathway) while leaving attention, MLPs, norms,
# token_refiner and the output heads on fl2va.
PRESETS: dict[str, list[tuple[Callable[[str], bool], bool]]] = {
    "none":                        [],   # pure base loading
    "ref2va_adaln_over_fl2va":     [(_match_block_adaln, True)],
    "ref2va_all_adaln_over_fl2va": [(_match_block_adaln, True), (_match_final_adaln, True)],
    "ref2va_full_over_fl2va":      [(_match_all, True)],
    "fl2va_full_over_ref2va":      [(_match_all, True)],
    # ``block_range_adaln`` is special-cased in load_minimax_h3_hybrid: the
    # actual block-range matcher is constructed there from the user's
    # block_range_start / block_range_end inputs. The empty list here is a
    # placeholder so PRESET_LIST / preset validation can recognise the name;
    # the live matchers are appended in load_minimax_h3_hybrid.
    "block_range_adaln":           [],
    "custom":                      [],   # filled in at runtime from custom_overlays
}

PRESET_LIST = list(PRESETS.keys())


# ---------------------------------------------------------------------------
# Core merging: build a state dict from two safetensors files
# ---------------------------------------------------------------------------

def _read_checkpoint_header(path: str):
    """Read the keys, metadata, layout, and tensor specs without retaining an mmap."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MiniMax H3 hybrid loader: file not found: {path}")
    with open(path, "rb") as header_f:
        header_size = struct.unpack("<Q", header_f.read(8))[0]
        header = json.loads(header_f.read(header_size))
    metadata = header.get("__metadata__") or {}
    tensor_header = {key: info for key, info in header.items() if key != "__metadata__"}
    keys = set(tensor_header)
    specs = {
        key: (info["shape"], info["dtype"])
        for key, info in tensor_header.items()
    }
    data_offset = 8 + header_size
    offsets = {
        key: (data_offset + info["data_offsets"][0],
              info["data_offsets"][1] - info["data_offsets"][0])
        for key, info in tensor_header.items()
    }
    topology, storage_format = _checkpoint_layout(keys, path, specs, offsets)
    return keys, metadata, topology, storage_format, specs, offsets


def _checkpoint_layout(keys: set[str], path: str, specs: dict, offsets: dict) -> tuple[str, str]:
    """Identify the ComfyUI MiniMax H3 topology and storage format."""
    required = {
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.weight",
    }
    missing = required - keys
    if missing:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: checkpoint is not a supported ComfyUI "
            f"MiniMax H3 diffusion model. missing={sorted(missing)} file={path!r}"
        )

    has_curve_table = "adaln_t_table" in keys
    has_time_embedder = "time_embedder.proj_in.weight" in keys
    if has_curve_table == has_time_embedder:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: cannot identify checkpoint topology; "
            "expected exactly one of adaln_t_table or time_embedder.proj_in.weight. "
            f"file={path!r}"
        )
    topology = "pruned" if has_curve_table else "full"

    quant_key = "blocks.0.attn.qkv_proj.comfy_quant"
    if quant_key in keys:
        try:
            offset, length = offsets[quant_key]
            with open(path, "rb") as quant_f:
                quant_f.seek(offset)
                quant_config = json.loads(quant_f.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"MiniMax H3 hybrid loader: invalid quantization metadata in {path!r}"
            ) from e
        quant_format = quant_config.get("format", "unknown")
        if quant_format == "int8_tensorwise" and quant_config.get("convrot", False):
            quant_format = "int8_convrot"
    else:
        quant_format = specs["blocks.0.attn.qkv_proj.weight"][1].lower()

    return topology, quant_format


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_TORCH_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F64": torch.float64,
}


def _spec_nbytes(spec: tuple[list[int], str]) -> int:
    shape, dtype = spec
    try:
        itemsize = _DTYPE_BYTES[dtype]
    except KeyError as e:
        raise RuntimeError(
            f"MiniMax H3 hybrid loader: unsupported safetensors dtype {dtype!r}"
        ) from e
    return prod(shape) * itemsize


def _map_tensor_slices_aimdo(path: str, keys: set[str], specs: dict, offsets: dict) -> dict:
    """Expose selected tensors through ComfyUI's managed ModelMMAP."""
    if not keys:
        return {}

    file_lock = threading.Lock()
    model_mmap = comfy_aimdo.model_mmap.ModelMMAP(path)
    file_handle = model_mmap.get_file_handle()
    file_size = os.path.getsize(path)
    file_view = memoryview((ctypes.c_uint8 * file_size).from_address(model_mmap.get()))
    out = {}
    for key in sorted(keys):
        shape, dtype = specs[key]
        absolute_offset, length = offsets[key]
        if length == 0:
            out[key] = torch.empty(shape, dtype=_TORCH_DTYPES[dtype])
            continue
        view = file_view[absolute_offset:absolute_offset + length]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            tensor = torch.frombuffer(view, dtype=_TORCH_DTYPES[dtype]).view(shape)
        storage = tensor.untyped_storage()
        setattr(storage, "_comfy_tensor_file_slice",
                comfy.memory_management.TensorFileSlice(
                    file_handle, file_lock, absolute_offset, length
                ))
        setattr(storage, "_comfy_tensor_mmap_refs", (model_mmap, file_view))
        out[key] = tensor
    return out


def _map_tensor_slices_fallback(path: str, keys: set[str], specs: dict, offsets: dict) -> dict:
    """Map selected byte ranges when AIMDO is unavailable."""
    if not keys:
        return {}

    out = {}
    file_handle = open(path, "rb")
    file_lock = threading.Lock()
    try:
        for key in sorted(keys):
            shape, dtype = specs[key]
            absolute_offset, length = offsets[key]
            if length == 0:
                out[key] = torch.empty(shape, dtype=_TORCH_DTYPES[dtype])
                continue
            aligned_offset = absolute_offset - (absolute_offset % mmap.ALLOCATIONGRANULARITY)
            view_offset = absolute_offset - aligned_offset
            mapped = mmap.mmap(
                file_handle.fileno(),
                view_offset + length,
                access=mmap.ACCESS_READ,
                offset=aligned_offset,
            )
            view = memoryview(mapped)[view_offset:view_offset + length]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                tensor = torch.frombuffer(view, dtype=_TORCH_DTYPES[dtype]).view(shape)
            storage = tensor.untyped_storage()
            setattr(storage, "_minimax_hybrid_mmap_refs",
                    (file_handle, mapped, view))
            setattr(storage, "_comfy_tensor_file_slice",
                    comfy.memory_management.TensorFileSlice(
                        file_handle, file_lock, absolute_offset, length
                    ))
            out[key] = tensor
    except Exception:
        file_handle.close()
        raise
    if not out:
        file_handle.close()
    return out


def _map_tensor_slices(path: str, keys: set[str], specs: dict, offsets: dict) -> dict:
    if comfy.memory_management.aimdo_enabled:
        return _map_tensor_slices_aimdo(path, keys, specs, offsets)
    return _map_tensor_slices_fallback(path, keys, specs, offsets)


def _expand_custom_overlay_globs(glob_csv: str | None) -> list[str]:
    """Parse the custom prefix / glob list. Supports both bare prefixes
    (e.g. ``blocks.3.``) and fnmatch globs (e.g. ``blocks.*.adaln_proj.``)."""
    if not glob_csv:
        return []
    out = []
    for raw in str(glob_csv).split(","):
        pat = raw.strip()
        if pat:
            out.append(pat)
    return out


def _match_glob_list(globs: list[str]) -> Callable[[str], bool]:
    if not globs:
        return lambda k: False

    def _m(k: str) -> bool:
        for g in globs:
            # Bare-prefix shorthand: a pattern with no fnmatch metacharacters
            # and ending in '.' is treated as a literal prefix.
            if g.endswith(".") and not any(c in g for c in "*?["):
                if k.startswith(g):
                    return True
            elif fnmatch.fnmatchcase(k, g):
                return True
        return False
    return _m


def build_hybrid_sd(
    base_path: str,
    overlay_path: str | None,
    overlay_groups: list[tuple[Callable[[str], bool], bool]],
) -> tuple[dict, dict]:
    """Build a merged state dict and metadata by streaming tensors from disk.

    Parameters
    ----------
    base_path
        Path to the "base" safetensors checkpoint. Every key starts here.
    overlay_path
        Path to the "overlay" checkpoint, or ``None``. May be the same as
        ``base_path`` (then overlay is a no-op, but we still validate keys).
    overlay_groups
        Ordered list of ``(matcher, take_from_overlay)``. The active overlay
        spec at evaluation time is the union of all matchers where
        ``take_from_overlay`` is True. ()

    Returns
    -------
    (sd, metadata)
        The merged state dict and base-checkpoint metadata, suitable for
        passing to ``comfy.sd.load_diffusion_model_state_dict``.

    Notes
    -----
    * With AIMDO enabled, each source uses one managed whole-file virtual
      mapping, while only selected tensors are exposed to the model. File pages
      remain demand-paged and participate in ComfyUI's normal bounce lifecycle.
      Header inspection uses ordinary file reads before either mapping exists.
    * The ``.comfy_quant`` byte tensors (which encode the shared int8 quant
      format) are present in both checkpoints and bit-identical between
      them, so taking them from either file is equivalent; they always
      travel with their underlying weight so we keep the choice consistent
      (i.e. a weight taken from the overlay file implies its
      ``.comfy_quant`` is also taken from the overlay file -- by virtue of
      the matcher applying to its ``.<weight|weight_scale>`` siblings only,
      the ``.comfy_quant`` of an overlay weight would, if not matched
      itself, come from the base. But since the two files' comfy_quant
      bytes are byte-identical, that is harmless. For safety, this loader
      always reads ``.comfy_quant`` from the *same* file as the weight it
      belongs to when the weight is part of an overlay group -- see below).
    """
    active_matchers = [m for (m, take) in overlay_groups if take]
    if not overlay_path:
        overlay_path = base_path
    if not active_matchers:
        overlay_path = base_path
    base_keys, base_metadata, base_topology, base_format, base_specs, base_offsets = _read_checkpoint_header(base_path)
    overlay_keys, overlay_metadata, overlay_topology, overlay_format, overlay_specs, overlay_offsets = _read_checkpoint_header(overlay_path)
    if base_topology != overlay_topology:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: pruned and full checkpoints cannot be mixed. "
            f"base={base_topology} overlay={overlay_topology}"
        )
    if base_format != overlay_format:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: checkpoints use different storage formats. "
            f"base={base_format} overlay={overlay_format}"
        )

    # Validate identical key sets -- required for a clean merge. The
    # header JSON of both files is identical (per our analysis), so any key
    # that's missing on one side indicates a corrupt / mismatched file
    # rather than a legitimate difference, and the merged model would fail
    # to load anyway with a far more cryptic error.
    only_base = base_keys - overlay_keys
    only_overlay = overlay_keys - base_keys
    if only_base or only_overlay:
        msg = ("MiniMax H3 hybrid loader: the two checkpoints have "
               "different key sets, cannot hybridise. base-only="
               f"{sorted(only_base)[:5]}{'...' if len(only_base)>5 else ''} "
               f"overlay-only={sorted(only_overlay)[:5]}"
               f"{'...' if len(only_overlay)>5 else ''}")
        raise RuntimeError(msg)

    incompatible = []
    for key in sorted(base_keys):
        if base_specs[key] != overlay_specs[key]:
            incompatible.append(key)
            if len(incompatible) == 5:
                break
    if incompatible:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: checkpoints contain incompatible tensor "
            f"shapes or dtypes. first mismatches={incompatible}"
        )

    base_config = base_metadata.get("config")
    overlay_config = overlay_metadata.get("config")
    if base_config and overlay_config and base_config != overlay_config:
        raise RuntimeError(
            "MiniMax H3 hybrid loader: checkpoints contain different model configs"
        )

    logging.info(
        "[MiniMaxH3Hybrid] detected topology=%s format=%s",
        base_topology,
        base_format,
    )

    # Compose the active overlay matcher: union of every group flagged True.
    # We *do not* allow a matcher's "take from base" (False) entry to retract
    # an earlier True; we only collect the True matchers and union them.
    def is_overlay_key(key: str) -> bool:
        return any(m(key) for m in active_matchers)

    # Sibling-resolution: when a quantized weight (e.g. ``attn.qkv_proj.weight``,
    # stored as int8) is matched by the overlay spec but the user's matcher
    # doesn't *also* match its int8 siblings (``.weight_scale``, ``.comfy_quant``),
    # we still pull those siblings from the overlay so the quantization metadata
    # co-travels with the weight it belongs to. The analysis shows these sibling
    # tensors are bit-identical across ref2va / fl2va, so this is correctness
    # hygiene rather than a correctness requirement -- but it's cheap and
    # eliminates a class of subtle bugs if a future checkpoint variant ships
    # different quant scales for an overlaid weight.

    def should_take_overlay(key: str) -> bool:
        take_overlay = is_overlay_key(key)
        if not take_overlay:
            # If this key is a quant sibling of some matched weight, infer its
            # provenance from the (matched) parent weight.
            if key.endswith(".comfy_quant"):
                parent = key[:-len(".comfy_quant")]
                if is_overlay_key(parent + ".weight") or is_overlay_key(parent):
                    take_overlay = True
            elif key.endswith("_scale"):
                parent = key[:-len("_scale")]
                if is_overlay_key(parent + ".weight") or is_overlay_key(parent):
                    take_overlay = True
        return take_overlay

    same_file = os.path.samefile(base_path, overlay_path)
    overlay_selected = (set() if same_file else
                        {key for key in base_keys if should_take_overlay(key)})
    base_selected = base_keys - overlay_selected
    overlay_bytes = sum(_spec_nbytes(overlay_specs[key]) for key in overlay_selected)
    base_bytes = sum(_spec_nbytes(base_specs[key]) for key in base_selected)

    # AIMDO uses one managed mapping per selected source and can bounce touched
    # pages at ComfyUI's normal execution boundaries. The fallback retains the
    # old per-tensor read-only mappings for systems without DynamicVRAM support.
    sd = _map_tensor_slices(base_path, base_selected, base_specs, base_offsets)
    sd.update(_map_tensor_slices(
        overlay_path, overlay_selected, overlay_specs, overlay_offsets
    ))
    sd = {key: sd[key] for key in sorted(sd)}

    logging.info(
        "[MiniMaxH3Hybrid] %s base=%.2f GiB/%d tensors overlay=%.2f GiB/%d tensors",
        "AIMDO-mapped" if comfy.memory_management.aimdo_enabled else "range-mapped",
        base_bytes / (1024 ** 3),
        len(base_selected),
        overlay_bytes / (1024 ** 3),
        len(overlay_selected),
    )

    return sd, base_metadata


# ---------------------------------------------------------------------------
# Hybrid loader factory (used both by the node and by cached_patcher_init
# so multigpu deepclone works)
# ---------------------------------------------------------------------------

def load_minimax_h3_hybrid(
    base_path: str,
    overlay_path: str | None,
    preset: str,
    custom_overlay_globs: str | None,
    custom_base_globs: str | None,
    weight_dtype: str = "default",
    block_range_start: int = 0,
    block_range_end: int = 49,
    final_adaln_from_overlay: bool = False,
    disable_dynamic: bool = False,
):
    """Build the merged state dict and load it as a ComfyUI MODEL.

    Returns a ``ModelPatcher`` exactly like ``comfy.sd.load_diffusion_model``.

    Preset handling
    ---------------
    * ``none`` / the four ``ref2va_*`` presets / ``custom`` are static.
    * ``block_range_adaln`` is dynamic: it overlays
      ``blocks.{block_range_start}..{block_range_end}.adaln_proj.linear.{weight,bias}``
      from the overlay, nothing else. Combine with ``final_adaln_from_overlay``
      to also pull the final-layer adaln.

    ``final_adaln_from_overlay`` is *additive*: it overlays
    ``final_layer.adaln_proj.linear.*`` on top of whatever the preset already
    does. So:

      * ``preset=ref2va_adaln_over_fl2va`` + ``final_adaln_from_overlay=True``
        is equivalent to ``preset=ref2va_all_adaln_over_fl2va`` (all blocks +
        final from ref2va).
      * ``preset=block_range_adaln`` + ``final_adaln_from_overlay=False``
        (the default) gives the "blocks 0..N adaln from ref2va, everything
        else (including final_layer.adaln) on fl2va" configuration the user
        described -- just by setting block_range_start / block_range_end.

    The two integer block-range inputs are read for any preset but only take
    effect when ``preset == "block_range_adaln"``; other adaln presets cover
    all 50 blocks regardless.
    """
    if preset not in PRESETS:
        raise RuntimeError(f"MiniMax H3 hybrid loader: unknown preset '{preset}'")

    overlay_groups: list[tuple[Callable[[str], bool], bool]] = []
    if preset == "block_range_adaln":
        # The range matcher takes effect in place of a static preset list.
        overlay_groups.append((
            _make_block_range_adaln_matcher(block_range_start, block_range_end),
            True))
    else:
        overlay_groups.extend(PRESETS[preset])

    # Additive final-layer toggle (independent of preset).
    if final_adaln_from_overlay:
        overlay_groups.append((_match_final_adaln, True))

    # Custom glob extensions (always applied on top of the preset):
    # these matchers add *more* keys to the overlay on top of the preset.
    if custom_overlay_globs:
        overlay_groups.append((
            _match_glob_list(_expand_custom_overlay_globs(custom_overlay_globs)), True))

    # ``custom_base_globs`` is a *retraction*: even if the preset or
    # custom_overlays would take a matching key from the overlay, force it
    # back to the base. We collapse the entire overlay spec into one
    # composite matcher (preset + custom_overlays, minus custom_base).
    if custom_base_globs:
        base_match = _match_glob_list(_expand_custom_overlay_globs(custom_base_globs))
        overlay_matchers = [m for (m, take) in overlay_groups if take]
        def composite_is_overlay(key: str) -> bool:
            if base_match(key):
                return False  # forced back to base
            return any(m(key) for m in overlay_matchers)
        overlay_groups = [(composite_is_overlay, True)]

    sd, metadata = build_hybrid_sd(base_path, overlay_path, overlay_groups)

    model_options: dict = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    model_patcher = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic
    )
    if model_patcher is None:
        raise RuntimeError(
            f"MiniMax H3 hybrid loader: ComfyUI could not detect the model type "
            f"from the merged state dict. Are both inputs actually MiniMax H3 "
            f"checkpoints? base={base_path!r} overlay={overlay_path!r} preset={preset!r}"
        )
    # Register the cached_patcher_init factory so ModelPatcher.deepclone_multigpu
    # and the disable_dynamic delegate path can reload a fresh copy from disk.
    # The factory is this same function called with the exact same arguments.
    model_patcher.cached_patcher_init = (
        load_minimax_h3_hybrid,
        (base_path, overlay_path, preset, custom_overlay_globs, custom_base_globs,
         weight_dtype, block_range_start, block_range_end, final_adaln_from_overlay,
         False),  # NOTE: disable_dynamic=False on reload so the deepcloned
                  # patcher is itself clonable.
    )
    return model_patcher


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

def _diffusion_model_filenames() -> list[str]:
    """List the contents of the 'diffusion_models' folder, sorted by
    modification time (the same convention UNETLoader uses)."""
    try:
        return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        # Fallback to alphabetical if a future ComfyUI renames the helper.
        return sorted(folder_paths.get_filename_list("diffusion_models", sort_by_modified=False))


class MiniMaxH3HybridLoader:
    """Hybrid loader for MiniMax H3 that overlays selected tensor groups from
    a second checkpoint onto a base checkpoint. Behaves exactly like the stock
    ``Load Diffusion Model`` node when ``overlay_preset == "none"``.

    See ``minimax_h3_analysis.md`` for the per-tensor comparison of
    ``ref2va`` vs ``fl2va`` and the rationale for the default preset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _diffusion_model_filenames()
        return {
            "required": {
                "base_model":      (files, {"tooltip": "Primary checkpoint -- every tensor starts here."}),
                "overlay_model":   (files, {"tooltip": "Secondary checkpoint -- tensors matched by overlay_preset come from here. May equal base_model (effectively loads base only)."}),
                "overlay_preset":  (PRESET_LIST, {"default": "block_range_adaln", "tooltip":
                    "Which tensor groups to take from overlay_model. "
                    "'none' = pure base loading (equivalent to stock UNETLoader). "
                    "'ref2va_adaln_over_fl2va' (default) = take per-block adaln_proj from the overlay only. "
                    "'ref2va_all_adaln_over_fl2va' = also take final_layer.adaln_proj. "
                    "'ref2va_full_over_fl2va'/'fl2va_full_over_ref2va' = take everything from the overlay (sanity check). "
                    "'block_range_adaln' = take adaln_proj only for blocks in [block_range_start, block_range_end] (inclusive). "
                    "'custom' = use custom_overlays only."}),
            },
            "optional": {
                "block_range_start":  ("INT", {"default": 25, "min": 0, "max": 49, "step": 1, "tooltip":
                    "Only used when overlay_preset == 'block_range_adaln'. "
                    "Lower-inclusive bound on the block index whose adaln_proj "
                    "comes from overlay_model. The minimax h3 DiT has 50 blocks "
                    "indexed 0..49."}),
                "block_range_end":    ("INT", {"default": 49, "min": 0, "max": 49, "step": 1, "tooltip":
                    "Only used when overlay_preset == 'block_range_adaln'. "
                    "Upper-inclusive bound on the block index. Set "
                    "block_range_end < block_range_start to take NO blocks "
                    "from the overlay (effectively pure base)."}),
                "final_adaln_from_overlay": ("BOOLEAN", {"default": False, "tooltip":
                    "Toggle overlay of final_layer.adaln_proj.linear.{weight,bias} "
                    "from overlay_model. Additive on top of any preset: "
                    " - False (default): leave final_layer.adaln on the base "
                    "(unless a preset already covers it, e.g. "
                    "ref2va_all_adaln_over_fl2va). "
                    " - True: pull final_layer.adaln from the overlay in "
                    "addition to whatever the preset does."}),
                "custom_overlays": ("STRING", {"multiline": False, "default": "", "tooltip":
                    "Comma-separated keys/prefixes/globs to *also* take from the overlay on top of the preset. "
                    "E.g. 'blocks.49.,final_layer.video_out.'. Bare prefixes ending in '.' match by prefix; "
                    "other strings are matched as fnmatch globs (e.g. 'blocks.[0-4].*.attn.qkv_proj.weight')."}),
                "custom_base":     ("STRING", {"multiline": False, "default": "", "tooltip":
                    "Comma-separated keys/prefixes/globs that should be forced *back* to the base even if "
                    "the preset or custom_overlays would take them from the overlay. Useful for "
                    "keeping e.g. final_layer.adaln_proj on the base while everything else adaln "
                    "comes from the overlay."}),
                "weight_dtype":    (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True, "tooltip":
                    "Same meaning as the stock 'Load Diffusion Model' node. Leave 'default' unless you know what you are doing."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_hybrid"
    CATEGORY = "model/loaders"
    DESCRIPTION = (
        "Load a MiniMax H3 diffusion model by merging selected tensor groups from "
        "an 'overlay' checkpoint onto a 'base' checkpoint. Useful for combining "
        "the higher output quality of fl2va with the reference-conditioning "
        "pathway of ref2va. See the companion analysis file for what differs "
        "between the two checkpoints. Behaves like the stock 'Load Diffusion "
        "Model' node when overlay_preset is 'none'."
    )

    def load_hybrid(self, base_model, overlay_model, overlay_preset,
                    block_range_start=0, block_range_end=49,
                    final_adaln_from_overlay=False,
                    custom_overlays="", custom_base="", weight_dtype="default"):
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        overlay_path = folder_paths.get_full_path_or_raise("diffusion_models", overlay_model)
        co = custom_overlays.strip() or None
        cb = custom_base.strip() or None
        patcher = load_minimax_h3_hybrid(
            base_path=base_path,
            overlay_path=overlay_path,
            preset=overlay_preset,
            custom_overlay_globs=co,
            custom_base_globs=cb,
            weight_dtype=weight_dtype,
            block_range_start=block_range_start,
            block_range_end=block_range_end,
            final_adaln_from_overlay=final_adaln_from_overlay,
        )
        # Log which combination was loaded so it shows up in the ComfyUI console.
        range_suffix = ""
        if overlay_preset == "block_range_adaln":
            range_suffix = f" blocks={block_range_start}..{block_range_end}"
        if final_adaln_from_overlay:
            range_suffix += " +final_adaln"
        display_preset = (overlay_preset if overlay_preset != "none"
                          else "none (= pure base, stock loader equivalent)")
        extra = []
        if co:
            extra.append(f"custom_overlays={co!r}")
        if cb:
            extra.append(f"custom_base={cb!r}")
        logging.info(
            "[MiniMaxH3Hybrid] base=%s overlay=%s preset=%s%s%s",
            os.path.basename(base_path),
            os.path.basename(overlay_path),
            display_preset,
            range_suffix,
            (" " + " ".join(extra)) if extra else "",
        )
        return (patcher,)

