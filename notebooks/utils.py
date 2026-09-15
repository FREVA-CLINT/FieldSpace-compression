"""Shared utilities for the hyperprior compression inference and training notebooks.

The functions in this module are deliberately small and side-effect free except
for the explicitly named artifact-writing helpers. Keeping them here leaves the
notebooks focused on the compression story while retaining reusable, testable
implementations of its nested-tensor and bookkeeping operations.
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict


@dataclass(frozen=True)
class InferenceSetup:
    """Validated paths, configuration, device, and derived experiment values."""

    profile: Mapping[str, Any]
    config_path: Path
    checkpoint_path: Path
    cfg: Any
    configured_variables: list[str]
    configured_variable_groups: Dict[str, list[str]]
    variable_group_levels: Dict[str, int]
    configured_zooms: list[int]
    device: torch.device
    max_zoom: int
    output_dir: Path


def _is_remote_location(location: Any) -> bool:
    """Return whether a data location uses a non-file URI scheme."""
    scheme = urlsplit(str(location)).scheme.lower()
    return bool(scheme and scheme != "file")


def _resolve_data_location(location: Any) -> str:
    """Resolve local data paths while preserving remote URLs verbatim."""
    value = str(location)
    if _is_remote_location(value):
        return value
    return str(Path(value).expanduser().resolve())


def split_variable_configuration(
    variables: Mapping[str, Any],
) -> Tuple[Dict[str, Mapping[str, Any]], Optional[Any]]:
    """Flatten grouped variable definitions and separate reserved timesteps."""
    groups, timesteps = grouped_variable_configuration(variables)
    specifications = {
        variable: specification
        for group in groups.values()
        for variable, specification in group.items()
    }
    return specifications, timesteps


def grouped_variable_configuration(
    variables: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Mapping[str, Any]]], Optional[Any]]:
    """Normalize ``2D``/``3D`` groups, retaining flat configs as 2-D shorthand."""
    if not isinstance(variables, Mapping):
        raise ValueError("VARIABLES must be a mapping.")
    entries = {str(key): value for key, value in variables.items() if str(key) != "timesteps"}
    uses_groups = any(group in entries for group in ("2D", "3D"))
    if uses_groups:
        unexpected = sorted(set(entries) - {"2D", "3D"})
        if unexpected:
            raise ValueError(
                "Grouped VARIABLES may contain only '2D', '3D', and 'timesteps'; "
                f"unexpected entries: {unexpected}."
            )
        groups: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        for group_name in ("2D", "3D"):
            group = entries.get(group_name)
            if group is None:
                continue
            if not isinstance(group, Mapping) or not group:
                raise ValueError(f"VARIABLES[{group_name!r}] must be a non-empty mapping.")
            groups[group_name] = {str(variable): spec for variable, spec in group.items()}
    else:
        # Backward compatibility for the original notebook schema.
        groups = {"2D": entries} if entries else {}

    names = [variable for group in groups.values() for variable in group]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Variables must belong to exactly one group; duplicates: {duplicates}.")
    return groups, variables.get("timesteps")


def expand_timestep_selection(
    selection: Any,
    *,
    n_timesteps: Optional[int] = None,
) -> list[int]:
    """Expand integer indices and end-exclusive ``"start-stop"`` ranges."""
    if not isinstance(selection, (list, tuple)) and not OmegaConf.is_list(selection):
        raise ValueError("`timesteps` must be a list of integers and/or 'start-stop' ranges.")
    expanded: list[int] = []
    for item in selection:
        if isinstance(item, int) and not isinstance(item, bool):
            if item < 0:
                raise ValueError(f"Timestep indices must be non-negative, got {item}.")
            expanded.append(int(item))
            continue
        if isinstance(item, str):
            match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", item)
            if match is None:
                raise ValueError(
                    f"Invalid timestep range {item!r}; use an end-exclusive range such as '5-100'."
                )
            start, stop = map(int, match.groups())
            if stop <= start:
                raise ValueError(f"Timestep range {item!r} must have stop > start.")
            expanded.extend(range(start, stop))
            continue
        raise ValueError(
            f"Unsupported timestep entry {item!r}; use integers or 'start-stop' strings."
        )
    if not expanded:
        raise ValueError("`timesteps` must select at least one index.")
    if len(set(expanded)) != len(expanded):
        raise ValueError("`timesteps` contains duplicate indices or overlapping ranges.")
    if n_timesteps is not None and max(expanded) >= int(n_timesteps):
        raise IndexError(
            f"Timestep index {max(expanded)} is outside the store with {n_timesteps} timesteps."
        )
    return expanded


def training_timestep_splits(
    variables: Mapping[str, Any], n_timesteps: int
) -> Dict[str, list[int]]:
    """Resolve an optional timestep pool and reserve its final 10+10 samples."""
    _, selection = split_variable_configuration(variables)
    pool = (
        list(range(int(n_timesteps)))
        if selection is None
        else expand_timestep_selection(selection, n_timesteps=int(n_timesteps))
    )
    if len(pool) < 21:
        raise ValueError(
            "At least 21 selected timesteps are required: one for training and 10 each "
            "for validation and testing."
        )
    return {"train": pool[:-20], "val": pool[-20:-10], "test": pool[-10:]}


def _normalize_level_indices(value: Any, variable: str) -> Optional[list[int]]:
    """Normalize an optional scalar/list level selection without resolving negatives."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
        indices = list(value)
        if not indices or any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise ValueError(f"level_indices for {variable!r} must contain one or more integers.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"level_indices for {variable!r} contains duplicates.")
        return [int(index) for index in indices]
    raise ValueError(f"level_indices for {variable!r} must be an integer, a list, or None.")


def _apply_inference_variable_overrides(
    cfg: Any,
    resolved_variable_groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Replace saved field-group selections with the inference profile selections.

    Replacing each complete field group is important when a profile omits
    ``level_indices``. In that case the inference store already contains the desired
    levels, and stale indices from the training config must not survive an OmegaConf
    dictionary merge. Non-field groups such as temporal embeddings are preserved.
    """
    for group_name, group in resolved_variable_groups.items():
        group_config = {
            variable: {"level_indices": specification.get("level_indices")}
            for variable, specification in group.items()
        }
        OmegaConf.update(
            cfg,
            f"data_split.test.variables.{group_name}",
            group_config,
            merge=False,
            force_add=True,
        )


def validate_inference_setup(
    model_profiles: Mapping[str, Mapping[str, Any]],
    model_profile: str,
    output_root: Path,
    device_policy: str,
    shard_size: int,
    num_workers: int,
) -> InferenceSetup:
    """Compose and validate an inference profile before the model is allocated."""
    from hydra import compose, initialize_config_dir

    if device_policy not in {"auto", "cuda", "cpu"}:
        raise ValueError("DEVICE_POLICY must be 'auto', 'cuda', or 'cpu'.")
    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size <= 0:
        raise ValueError("SHARD_SIZE must be a positive integer.")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("NUM_WORKERS must be a non-negative integer.")
    if model_profile not in model_profiles:
        raise ValueError(
            f"Unknown MODEL_PROFILE {model_profile!r}; choose from {sorted(model_profiles)}."
        )

    profile = model_profiles[model_profile]
    required_fields = {
        "label",
        "config_path",
        "checkpoint_path",
        "variables",
        "expected_zooms",
    }
    missing_fields = required_fields.difference(profile)
    if missing_fields:
        raise ValueError(f"Profile {model_profile!r} is missing fields: {sorted(missing_fields)}")

    config_path = Path(profile["config_path"]).expanduser().resolve()
    checkpoint_path = Path(profile["checkpoint_path"]).expanduser().resolve()
    for label, required_path in (("config", config_path), ("checkpoint", checkpoint_path)):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing {label}: {required_path}")

    with initialize_config_dir(
        config_dir=str(config_path.parent), job_name="hackathon_inference", version_base=None
    ):
        cfg = compose(config_name=config_path.stem)

    configured_variable_groups = {
        str(group_name): [str(variable) for variable in group]
        for group_name, group in cfg.data_split.test.variables.items()
        if str(group_name) not in {"embedding", "embedding_1D"}
    }
    configured_variables = [
        variable for group in configured_variable_groups.values() for variable in group
    ]
    configured_zooms = sorted(int(zoom) for zoom in cfg.model.model.in_zooms)
    profile_variables = profile["variables"]
    profile_variable_groups, timestep_selection = grouped_variable_configuration(profile_variables)
    variable_specs = {
        variable: specification
        for group in profile_variable_groups.values()
        for variable, specification in group.items()
    }
    if not variable_specs:
        raise ValueError(f"Profile {model_profile!r} must define a non-empty variables mapping.")
    selected_variable_groups = {
        group: list(group_variables) for group, group_variables in profile_variable_groups.items()
    }
    expected_zooms = sorted(int(zoom) for zoom in profile["expected_zooms"])
    if configured_variable_groups != selected_variable_groups:
        raise ValueError(
            "Variable-group mismatch: "
            f"config={configured_variable_groups}, profile={selected_variable_groups}"
        )
    if configured_zooms != expected_zooms:
        raise ValueError(f"Zoom mismatch: config={configured_zooms}, profile={expected_zooms}")
    # A profile points each variable to its highest-resolution inference store.
    resolved_variable_groups: Dict[str, Dict[str, Any]] = {}
    for group_name, group_variables in profile_variable_groups.items():
        resolved_variable_groups[group_name] = {}
        for variable, specification in group_variables.items():
            if not isinstance(specification, Mapping) or not specification.get("path"):
                raise ValueError(f"Profile variable {variable!r} needs a non-empty path.")
            _normalize_level_indices(specification.get("level_indices"), variable)
            resolved_variable_groups[group_name][str(variable)] = {
                "path": _resolve_data_location(specification["path"]),
                "level_indices": specification.get("level_indices"),
            }
    resolved_variables = {
        variable: specification
        for group in resolved_variable_groups.values()
        for variable, specification in group.items()
    }

    # Validate variable presence, level selection, alignment, and max-grid resolution.
    inspection = inspect_variable_stores(
        resolved_variable_groups,
        expected_zoom=max(configured_zooms),
        minimum_timesteps=1,
    )
    if timestep_selection is None:
        # Ignore the split stored in the training config and let the dataset use
        # every valid timestep in the inference stores.
        with open_dict(cfg.data_split.test):
            cfg.data_split.test.pop("timesteps", None)
    else:
        cfg.data_split.test.timesteps = expand_timestep_selection(
            timestep_selection, n_timesteps=inspection.n_timesteps
        )

    group_names = list(inspection.variable_groups)
    expected_group_variables = [len(inspection.variable_groups[group]) for group in group_names]
    expected_group_depths = [1] * len(group_names)
    expected_token_depths = [inspection.group_levels[group] for group in group_names]
    if list(cfg.model.model.n_groups_variables) != expected_group_variables:
        raise ValueError(
            f"n_groups_variables mismatch: config={list(cfg.model.model.n_groups_variables)}, "
            f"data={expected_group_variables}."
        )
    if int(cfg.embedding.MGEmbedder.n_variables) != len(configured_variables):
        raise ValueError(
            f"Embedding n_variables mismatch: config={cfg.embedding.MGEmbedder.n_variables}, "
            f"data={len(configured_variables)}."
        )
    if list(cfg.model.model.n_groups_depths) != expected_group_depths:
        raise ValueError(
            f"n_groups_depths mismatch: config={list(cfg.model.model.n_groups_depths)}, "
            f"expected={expected_group_depths}."
        )
    for blocks in (
        cfg.model.model.analysis_block_configs,
        cfg.model.model.synthesis_block_configs,
        cfg.model.model.hyper_analysis_block_configs,
        cfg.model.model.hyper_synthesis_block_configs,
    ):
        for block in blocks.values():
            if "att_dim" in block and list(block.token_len_depth) != expected_token_depths:
                raise ValueError(
                    f"token_len_depth mismatch: config={list(block.token_len_depth)}, "
                    f"data={expected_token_depths}."
                )

    # Repeat one max-resolution anchor under every zoom; variable_files directs
    # the loader to each variable's actual store while lower zooms are derived lazily.
    anchor_path = str(resolved_variables[configured_variables[0]]["path"])
    for section in ("source", "target"):
        for zoom_config in cfg.data_split.test[section].values():
            zoom_config.files = [anchor_path]
    OmegaConf.update(
        cfg,
        "data_split.test.variable_files",
        {
            variable: {"files": [str(specification["path"])]}
            for variable, specification in resolved_variables.items()
        },
        force_add=True,
    )
    # The profile is authoritative: an omitted/None level selection means that all
    # levels in the inference store are used, rather than the saved training indices.
    _apply_inference_variable_overrides(cfg, resolved_variable_groups)

    # Check every configured test input, including the optional per-variable store schema.
    data_locations: list[tuple[str, str]] = []
    for section in ("source", "target"):
        for zoom_config in cfg.data_split.test[section].values():
            data_locations.extend(
                (f"{section} data", _resolve_data_location(path))
                for path in zoom_config.files
            )
    variable_files = cfg.data_split.test.get("variable_files")
    if variable_files:
        for variable, variable_config in variable_files.items():
            data_locations.extend(
                (f"data for {variable}", _resolve_data_location(path))
                for path in variable_config.files
            )
    for label, data_location in data_locations:
        # Remote stores were already opened by inspect_variable_stores; only
        # local paths have a meaningful pathlib existence check.
        if not _is_remote_location(data_location) and not Path(data_location).exists():
            raise FileNotFoundError(f"Missing {label}: {data_location}")

    norm_path = Path(cfg.dataloader.dataset.norm_dict).expanduser()
    if not norm_path.is_file():
        raise FileNotFoundError(f"Missing normalizer: {norm_path}")

    if device_policy == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('DEVICE_POLICY="cuda" but no CUDA device is available.')
    if device_policy == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_zoom = max(configured_zooms)
    output_dir = (Path(output_root) / model_profile).expanduser().resolve()
    return InferenceSetup(
        profile=profile,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        configured_variables=configured_variables,
        configured_variable_groups=configured_variable_groups,
        variable_group_levels=inspection.group_levels,
        configured_zooms=configured_zooms,
        device=device,
        max_zoom=max_zoom,
        output_dir=output_dir,
    )


def inference_experiment_summary(experiment: InferenceSetup) -> Any:
    """Build the compact profile/configuration summary displayed by inference."""
    profile_groups, _ = grouped_variable_configuration(experiment.profile["variables"])
    test_timesteps = experiment.cfg.data_split.test.get("timesteps")
    variable_paths = "; ".join(
        f"{variable}: {specification['path']}"
        for group in profile_groups.values()
        for variable, specification in group.items()
    )
    return pd.Series(
        {
            "Profile": experiment.profile["label"],
            "Variables": ", ".join(experiment.configured_variables),
            "Variable groups": "; ".join(
                f"{group}: {', '.join(variables)}"
                for group, variables in experiment.configured_variable_groups.items()
            ),
            "Variable data paths": variable_paths,
            "Zooms": ", ".join(map(str, experiment.configured_zooms)),
            "Configured test split": (
                ", ".join(map(str, test_timesteps))
                if test_timesteps is not None
                else "all available timesteps"
            ),
            "Device": str(experiment.device),
            "Checkpoint size (GiB)": f"{experiment.checkpoint_path.stat().st_size / 2**30:.2f}",
            "Output directory": str(experiment.output_dir),
        },
        name="value",
    ).to_frame()


def find_repo_root(start: Path) -> Path:
    """Find the nearest parent containing the FieldSpaceNN source and snapshots."""
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "src" / "FieldSpaceNN").is_dir() and (candidate / "snapshots").is_dir():
            return candidate
    raise RuntimeError("Could not locate the FieldSpaceNN repository root.")


def collate_dataset_sample(dataset: Any, collator: Callable[[Any], Any], dataset_index: int) -> Any:
    """Load and collate one dataset item, preserving the notebook's batch-size-one contract."""
    return collator([dataset[dataset_index]])


def tree_map(value: Any, tensor_fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply a function to every tensor in a nested Python container."""
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: tree_map(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_map(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_map(item, tensor_fn) for item in value)
    return value


def to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested container to ``device``."""
    return tree_map(value, lambda tensor: tensor.to(device, non_blocking=True))


def to_cpu(value: Any) -> Any:
    """Detach every nested tensor and move it to CPU for serialization or metrics."""
    return tree_map(value, lambda tensor: tensor.detach().cpu())


def indexed_item(value: Any, index: int) -> Any:
    """Select one batch item recursively while preserving container structure."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value[index]
    if isinstance(value, dict):
        return {key: indexed_item(item, index) for key, item in value.items()}
    if isinstance(value, list):
        return [indexed_item(item, index) for item in value]
    if isinstance(value, tuple):
        return tuple(indexed_item(item, index) for item in value)
    return value


def as_group_sequence(value: Any) -> Any:
    """Wrap a group dictionary in the sequence expected by the codec adapters."""
    return [value] if isinstance(value, dict) else value


def first_group(value: Any) -> Any:
    """Return the first codec group, retaining ``None`` for absent side information."""
    return None if value is None else value[0]


def effective_sample_configs(
    dataset: Any,
    codec: Any,
    patch_index_zooms: Mapping[int, torch.Tensor],
) -> Dict[int, Dict[str, Any]]:
    """Build the effective sampling configuration for a manually invoked codec batch."""
    # Imported lazily because the notebook discovers and adds the package root
    # after importing this local utility module.
    from fieldspacenn.src.utils.helpers import merge_sampling_dicts

    source = dataset.sampling_zooms_collate or dataset.sampling_zooms
    if OmegaConf.is_config(source):
        source = OmegaConf.to_container(source, resolve=True)
    configs = {int(zoom): dict(values) for zoom, values in source.items()}
    configs = merge_sampling_dicts(configs, patch_index_zooms)
    for zoom in codec.in_zooms:
        zoom_config = dict(configs.get(int(zoom), {}) or {})
        zoom_config.setdefault("n_past_ts", 0)
        zoom_config.setdefault("n_future_ts", 0)
        zoom_config.setdefault("zoom_patch_sample", -1)
        zoom_config.setdefault("patch_index", 0)
        configs[int(zoom)] = zoom_config
    return configs


def merge_time_side_info(embedding_groups: Any, side_info: Any) -> Any:
    """Add time embeddings that are required for standalone decompression."""
    if not isinstance(embedding_groups, list) or side_info is None:
        return side_info
    merged = []
    for embedding_group, metadata_group in zip(embedding_groups, side_info):
        current = dict(metadata_group or {})
        if isinstance(embedding_group, dict) and embedding_group.get("TimeEmbedder") is not None:
            current["TimeEmbedder"] = to_cpu(embedding_group["TimeEmbedder"])
        merged.append(current)
    return merged


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA timings while remaining a no-op for CPU inference."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def json_ready(value: Any) -> Any:
    """Convert paths, NumPy scalars, tensors, and nested containers to JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def dataset_time_index(dataset: Any, max_zoom: int, dataset_index: int) -> int:
    """Return the physical time index recorded in a dataset's max-zoom index map."""
    row = dataset.index_map[max_zoom][dataset_index]
    return int(row[2])


def clean_generated_outputs(
    output_dir: Path,
    preserve: Sequence[str] = (),
) -> None:
    """Remove known notebook artifacts, optionally retaining selected filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve_names = {str(name) for name in preserve}
    known_names = {
        "manifest.json",
        "metrics_per_variable.csv",
        "metrics_summary.json",
        "reconstruction_maps.png",
        "deep_diagnostics.png",
        "rmse_vs_compression.png",
        "input_zoom_preview.png",
        "input_zoom_decomposition.png",
    }
    for path in output_dir.iterdir():
        is_shard = path.name.startswith("compressed_shard_") and path.suffix == ".pt"
        should_remove = path.name in known_names or is_shard
        if path.is_file() and should_remove and path.name not in preserve_names:
            path.unlink()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a human-readable JSON artifact after converting non-JSON values."""
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


_COMPACT_SHARD_FORMAT = "fieldspacenn_compact_shard_v1"


def _all_equal(values: Sequence[Any]) -> bool:
    """Return whether non-tensor metadata values can be stored once."""
    first = values[0]
    for value in values[1:]:
        if type(value) is not type(first):
            return False
        try:
            equal = value == first
            if isinstance(equal, (bool, np.bool_)):
                if not bool(equal):
                    return False
            else:
                return False
        except Exception:
            return False
    return True


def _pack_compact_values(values: Sequence[Any]) -> Dict[str, Any]:
    """Pack corresponding record values into a compact, reversible tree."""
    if not values:
        raise ValueError("Cannot compact an empty value sequence.")
    first = values[0]

    if all(torch.is_tensor(value) for value in values):
        tensors = [value.detach().cpu() for value in values]
        same_layout = all(
            tensor.shape == tensors[0].shape and tensor.dtype == tensors[0].dtype
            for tensor in tensors[1:]
        )
        if same_layout and all(torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
            tensor = tensors[0].contiguous()
            return {
                "k": "tensor_static",
                "v": tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
                "s": tuple(tensor.shape),
                "d": tensor.dtype,
            }
        if same_layout:
            # Batch the field, then store raw bytes inside the pickle payload. This
            # avoids a separate PyTorch ZIP storage entry for every small tensor.
            tensor = torch.stack(tensors, dim=0).contiguous()
            return {
                "k": "tensor_batch",
                "v": tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
                "s": tuple(tensor.shape),
                "d": tensor.dtype,
            }
        return {"k": "items", "v": tensors}

    if all(isinstance(value, (bytes, bytearray)) for value in values):
        chunks = [bytes(value) for value in values]
        offsets = [0]
        for chunk in chunks:
            offsets.append(offsets[-1] + len(chunk))
        # Entropy strings sharing a tree position become one byte stream.
        return {"k": "bytes", "v": b"".join(chunks), "o": offsets}

    if all(isinstance(value, dict) for value in values):
        keys = list(first)
        if all(list(value) == keys for value in values[1:]):
            return {
                "k": "dict",
                "v": [(key, _pack_compact_values([value[key] for value in values])) for key in keys],
            }

    if all(isinstance(value, list) for value in values):
        length = len(first)
        if all(len(value) == length for value in values[1:]):
            return {
                "k": "list",
                "v": [_pack_compact_values([value[index] for value in values]) for index in range(length)],
            }

    if all(isinstance(value, tuple) for value in values):
        length = len(first)
        if all(len(value) == length for value in values[1:]):
            return {
                "k": "tuple",
                "v": [_pack_compact_values([value[index] for value in values]) for index in range(length)],
            }

    if _all_equal(values):
        return {"k": "static", "v": first}
    return {"k": "items", "v": list(values)}


def _compact_tensor(node: Dict[str, Any]) -> torch.Tensor:
    """Materialize and cache a tensor stored as compact raw bytes."""
    if "_decoded" not in node:
        shape = tuple(int(size) for size in node["s"])
        if math.prod(shape) == 0:
            tensor = torch.empty(shape, dtype=node["d"])
        else:
            # bytearray provides writable storage retained by the resulting tensor.
            tensor = torch.frombuffer(bytearray(node["v"]), dtype=node["d"]).reshape(shape)
        node["_decoded"] = tensor
    return node["_decoded"]


def _unpack_compact_value(node: Dict[str, Any], index: int) -> Any:
    """Reconstruct one record value from a compact tree."""
    kind = node["k"]
    if kind == "static":
        return node["v"]
    if kind == "tensor_static":
        return _compact_tensor(node)
    if kind == "tensor_batch":
        return _compact_tensor(node)[index]
    if kind == "bytes":
        start, stop = node["o"][index : index + 2]
        return node["v"][start:stop]
    if kind == "items":
        return node["v"][index]
    if kind == "dict":
        return {key: _unpack_compact_value(value, index) for key, value in node["v"]}
    if kind == "list":
        return [_unpack_compact_value(value, index) for value in node["v"]]
    if kind == "tuple":
        return tuple(_unpack_compact_value(value, index) for value in node["v"])
    raise ValueError(f"Unknown compact shard node kind: {kind!r}.")


def _compact_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Create the versioned compact representation of a record shard."""
    if not records:
        raise ValueError("Cannot write an empty compressed shard.")
    return {
        "format": _COMPACT_SHARD_FORMAT,
        "record_count": len(records),
        "record_tree": _pack_compact_values(records),
    }


def load_shard(shard_path: Path) -> list[Dict[str, Any]]:
    """Load compact or legacy compressed records through one stable interface."""
    payload = torch.load(Path(shard_path), map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and payload.get("format") == _COMPACT_SHARD_FORMAT):
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported compressed shard payload in {shard_path}.")
        return payload
    count = int(payload["record_count"])
    if count <= 0:
        raise ValueError(f"Compact shard {shard_path} contains no records.")
    return [_unpack_compact_value(payload["record_tree"], index) for index in range(count)]


def write_shard(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    shard_index: int,
    compact_artifact_format: bool = True,
) -> Tuple[Path, float]:
    """Serialize one compact or legacy record shard and return path/write time."""
    shard_path = output_dir / f"compressed_shard_{shard_index:04d}.pt"
    payload: Any = _compact_records(records) if compact_artifact_format else list(records)
    started = time.perf_counter()
    if compact_artifact_format:
        # Protocol 5 writes concatenated binary streams directly. Protocol 2,
        # PyTorch's default, expands arbitrary bytes substantially in data.pkl.
        torch.save(payload, shard_path, pickle_protocol=5)
    else:
        torch.save(payload, shard_path)
    write_seconds = time.perf_counter() - started
    return shard_path, write_seconds


def metrics_from_accumulators(accumulators: Mapping[str, Mapping[str, float]]) -> Any:
    """Convert streaming error sums into per-variable and pooled metrics."""
    import pandas as pd

    metric_rows = []
    for variable, stats in accumulators.items():
        count = int(stats["count"])
        if count <= 0:
            raise ValueError(f"No values were accumulated for {variable!r}.")
        target_variance = stats["sum_target_sq"] - stats["sum_target"] ** 2 / count
        prediction_variance = (
            stats["sum_prediction_sq"] - stats["sum_prediction"] ** 2 / count
        )
        covariance = stats["sum_cross"] - stats["sum_target"] * stats["sum_prediction"] / count
        correlation = covariance / math.sqrt(max(target_variance * prediction_variance, 1e-30))
        metric_rows.append(
            {
                "variable": variable,
                "values": count,
                "RMSE": math.sqrt(stats["sum_squared_error"] / count),
                "MAE": stats["sum_abs_error"] / count,
                "bias": stats["sum_error"] / count,
                "Pearson r": correlation,
            }
        )

    total_count = sum(int(stats["count"]) for stats in accumulators.values())
    if total_count <= 0:
        raise ValueError("No reconstruction values were accumulated.")
    total_sse = sum(stats["sum_squared_error"] for stats in accumulators.values())
    total_sae = sum(stats["sum_abs_error"] for stats in accumulators.values())
    total_error = sum(stats["sum_error"] for stats in accumulators.values())
    metric_rows.append(
        {
            "variable": "TOTAL (pooled)",
            "values": total_count,
            "RMSE": math.sqrt(total_sse / total_count),
            "MAE": total_sae / total_count,
            "bias": total_error / total_count,
            "Pearson r": np.nan,
        }
    )
    metrics = pd.DataFrame(metric_rows)
    if not np.isfinite(metrics[["RMSE", "MAE", "bias"]].to_numpy()).all():
        raise ValueError("Non-finite reconstruction metrics detected.")
    return metrics


def build_scorecard(
    *,
    sample_count: int,
    raw_hpx_bytes: int,
    raw_hpx_values: int,
    artifact_bytes: int,
    model_load_seconds: float,
    compression_seconds: float,
    decompression_seconds: float,
    artifact_write_seconds: float,
) -> Dict[str, float]:
    """Calculate rate, timing, and throughput values used by the dashboard."""
    if raw_hpx_values <= 0 or raw_hpx_bytes <= 0 or artifact_bytes <= 0:
        raise ValueError("Raw value, raw byte, and artifact byte counts must be positive.")
    raw_mib = raw_hpx_bytes / 2**20
    return {
        "samples": int(sample_count),
        "raw_hpx9_MiB": raw_mib,
        "artifact_MiB": artifact_bytes / 2**20,
        "compression_ratio_raw_hpx9_over_artifact": raw_hpx_bytes / artifact_bytes,
        "artifact_bits_per_hpx9_value": 8 * artifact_bytes / raw_hpx_values,
        "model_load_seconds": float(model_load_seconds),
        "compression_seconds": float(compression_seconds),
        "decompression_seconds": float(decompression_seconds),
        "artifact_write_seconds": float(artifact_write_seconds),
        "compression_throughput_raw_MiB_per_s": raw_mib / max(compression_seconds, 1e-12),
        "decompression_throughput_raw_MiB_per_s": raw_mib / max(decompression_seconds, 1e-12),
    }


def save_evaluation_summary(
    output_dir: Path,
    metrics: Any,
    scorecard: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write dashboard tables and return the manifest augmented with evaluation metadata."""
    output_dir = Path(output_dir)
    metrics.to_csv(output_dir / "metrics_per_variable.csv", index=False)
    write_json(output_dir / "metrics_summary.json", scorecard)
    updated_manifest = dict(manifest)
    updated_manifest.update(
        {
            "decompression_seconds": scorecard["decompression_seconds"],
            "metrics_files": ["metrics_per_variable.csv", "metrics_summary.json"],
        }
    )
    write_json(output_dir / "manifest.json", updated_manifest)
    return updated_manifest


def _decoded_record_time_indices(
    dataset: Any,
    record: Mapping[str, Any],
    max_zoom: int,
    decoded_time_count: int,
) -> list[int]:
    """Recover source time indices represented by one decoded artifact record."""
    dataset_index = int(record["dataset_index"])
    row = dataset.index_map[int(max_zoom)][dataset_index]
    centers = [int(value) for value in row[2:]]
    sample_configs = record["metadata"]["sample_configs"]
    zoom_config = sample_configs.get(int(max_zoom), sample_configs.get(str(max_zoom)))
    if zoom_config is None:
        raise KeyError(f"Artifact sample configuration has no HPX{max_zoom} entry.")
    n_past = int(zoom_config.get("n_past_ts", 0))
    n_future = int(zoom_config.get("n_future_ts", 0))
    indices = [
        index
        for center in centers
        for index in range(center - n_past, center + n_future + 1)
    ]
    if len(indices) != int(decoded_time_count):
        raise ValueError(
            "Decoded time axis does not match the dataset sample window: "
            f"tensor={decoded_time_count}, source indices={indices}."
        )
    return indices


def _decoded_variable_array(
    values: np.ndarray,
    source_array: xr.DataArray,
    specification: Mapping[str, Any],
    time_indices: Sequence[int],
    variable: str,
) -> Tuple[xr.DataArray, Optional[str], Optional[list[int]]]:
    """Attach one decoded ``(time, cell, depth)`` array to source metadata."""
    spatial_dim = _spatial_dimension(source_array)
    level_dims = [
        dim for dim in source_array.dims if dim not in ("time", spatial_dim)
    ]
    if len(level_dims) > 1:
        raise ValueError(
            f"Variable `{variable}` has unsupported source dimensions {source_array.dims}."
        )
    level_dim = level_dims[0] if level_dims else None
    level_indices = _normalize_level_indices(
        specification.get("level_indices"), variable
    )
    selectors: Dict[str, Any] = {"time": list(map(int, time_indices))}
    if level_dim is not None and level_indices is not None:
        selectors[level_dim] = level_indices
    selected = source_array.isel(selectors)

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            f"Decoded `{variable}` must have (time, cell, depth) shape, got {values.shape}."
        )
    expected_cells = int(selected.sizes[spatial_dim])
    expected_depth = 1 if level_dim is None else int(selected.sizes[level_dim])
    expected_shape = (len(time_indices), expected_cells, expected_depth)
    if values.shape != expected_shape:
        raise ValueError(
            f"Decoded `{variable}` shape {values.shape} does not match source metadata "
            f"shape {expected_shape}."
        )

    if level_dim is None:
        decoded = xr.DataArray(
            values[..., 0], dims=("time", spatial_dim), name=variable
        )
    else:
        decoded = xr.DataArray(
            values, dims=("time", spatial_dim, level_dim), name=variable
        ).transpose(*selected.dims)
    decoded = decoded.assign_coords(
        {
            name: coordinate
            for name, coordinate in selected.coords.items()
            if set(coordinate.dims).issubset(decoded.dims)
        }
    )
    decoded.attrs = copy.deepcopy(source_array.attrs)
    if "_FillValue" in source_array.encoding:
        decoded.encoding["_FillValue"] = source_array.encoding["_FillValue"]
    return decoded, level_dim, level_indices


def _attach_grid_mapping_metadata(
    output: xr.Dataset,
    source: xr.Dataset,
    source_array: xr.DataArray,
    time_indices: Sequence[int],
    level_dim: Optional[str],
    level_indices: Optional[Sequence[int]],
) -> xr.Dataset:
    """Copy CF grid-mapping variables referenced by a reconstructed field."""
    raw_mapping = source_array.attrs.get("grid_mapping")
    if not raw_mapping:
        return output
    candidates = [
        token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(raw_mapping))
        if token in source.variables
    ]
    for name in candidates:
        if name in output.variables:
            continue
        metadata = source[name]
        selectors: Dict[str, Any] = {}
        if "time" in metadata.dims:
            selectors["time"] = list(map(int, time_indices))
        if level_dim is not None and level_dim in metadata.dims and level_indices is not None:
            selectors[level_dim] = list(level_indices)
        if selectors:
            metadata = metadata.isel(selectors)
        if name in source.coords:
            output = output.assign_coords({name: metadata})
        else:
            output[name] = metadata
    return output


def export_decoded_zarr(
    *,
    output_path: Path,
    manifest: Mapping[str, Any],
    experiment: InferenceSetup,
    dataset: Any,
    codec: Any,
    overwrite: bool = False,
    spatial_chunk_size: int = 65_536,
) -> Dict[str, Any]:
    """Stream denormalized decoded fields to a metadata-preserving Zarr store."""
    from fieldspacenn.src.modules.grids.grid_utils import decode_zooms
    from tqdm.auto import tqdm
    import zarr

    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix != ".zarr":
        raise ValueError("Decoded output path must end in `.zarr`.")
    if spatial_chunk_size <= 0:
        raise ValueError("spatial_chunk_size must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Decoded Zarr already exists: {output_path}. Set overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = output_path.with_name(
        f".{output_path.name}.tmp-{time.time_ns()}"
    )

    profile_groups, _ = grouped_variable_configuration(
        experiment.profile["variables"]
    )
    specifications = {
        variable: specification
        for group in profile_groups.values()
        for variable, specification in group.items()
    }
    locations = {
        variable: _resolve_data_location(specification["path"])
        for variable, specification in specifications.items()
    }
    source_datasets: Dict[str, xr.Dataset] = {}
    seen_time_indices: set[int] = set()
    written_records = 0
    written_times = 0
    first_write = True

    try:
        for location in dict.fromkeys(locations.values()):
            source_datasets[location] = xr.open_dataset(
                location, decode_times=False, create_default_indexes=False
            )
        source_global_attrs = {
            location: json_ready(source.attrs)
            for location, source in source_datasets.items()
        }
        anchor_attrs = copy.deepcopy(next(iter(source_datasets.values())).attrs)
        normalizer_zoom = max(int(zoom) for zoom in dataset.var_normalizers)

        progress = tqdm(
            total=int(manifest["sample_count"]),
            desc="Exporting decoded Zarr",
            unit="sample",
        )
        try:
            for shard_info in manifest["shards"]:
                records = load_shard(experiment.output_dir / shard_info["name"])
                for record in records:
                    metadata_device = to_device(record["metadata"], experiment.device)
                    strings_device = to_device(record["strings"], experiment.device)
                    with torch.inference_mode():
                        decoded = codec.decompress(
                            strings_device,
                            metadata_device,
                            sample_configs=metadata_device["sample_configs"],
                            out_zoom=experiment.max_zoom,
                        )

                    decoded_arrays: Dict[str, xr.DataArray] = {}
                    grid_metadata: list[
                        Tuple[xr.Dataset, xr.DataArray, Optional[str], Optional[list[int]]]
                    ] = []
                    record_time_indices: Optional[list[int]] = None
                    for group_index, (group_name, group_variables) in enumerate(
                        experiment.configured_variable_groups.items()
                    ):
                        decoded_group = decode_zooms(
                            to_cpu(decoded["x_hat"][group_index]),
                            sample_configs=record["metadata"]["sample_configs"],
                            out_zoom=experiment.max_zoom,
                        )[experiment.max_zoom].float()
                        if decoded_group.ndim != 6 or decoded_group.shape[0] != 1 or decoded_group.shape[-1] != 1:
                            raise ValueError(
                                "Decoded tensors must have shape (1, variable, time, cell, depth, 1); "
                                f"got {tuple(decoded_group.shape)} for group {group_name!r}."
                            )
                        if decoded_group.shape[1] != len(group_variables):
                            raise ValueError(
                                f"Decoded variable count does not match group {group_name!r}."
                            )
                        current_time_indices = _decoded_record_time_indices(
                            dataset,
                            record,
                            experiment.max_zoom,
                            int(decoded_group.shape[2]),
                        )
                        if record_time_indices is None:
                            record_time_indices = current_time_indices
                        elif record_time_indices != current_time_indices:
                            raise ValueError("Decoded groups have inconsistent time axes.")

                        for variable_index, variable in enumerate(group_variables):
                            normalizer = dataset.var_normalizers[normalizer_zoom][variable]
                            physical = normalizer.denormalize(
                                decoded_group[:, variable_index : variable_index + 1]
                            )
                            values = physical[0, 0, ..., 0].numpy()
                            source = source_datasets[locations[variable]]
                            source_array = source[variable]
                            data_array, level_dim, level_indices = _decoded_variable_array(
                                values,
                                source_array,
                                specifications[variable],
                                current_time_indices,
                                variable,
                            )
                            decoded_arrays[variable] = data_array
                            grid_metadata.append(
                                (source, source_array, level_dim, level_indices)
                            )

                    assert record_time_indices is not None
                    overlap = seen_time_indices.intersection(record_time_indices)
                    if overlap:
                        raise ValueError(
                            "Decoded sample windows overlap at source time indices "
                            f"{sorted(overlap)}; a unique time axis is required for Zarr export."
                        )
                    seen_time_indices.update(record_time_indices)

                    piece = xr.Dataset(decoded_arrays, attrs=copy.deepcopy(anchor_attrs))
                    piece = piece.assign_coords(
                        source_time_index=(
                            "time",
                            np.asarray(record_time_indices, dtype=np.int64),
                        )
                    )
                    piece["source_time_index"].attrs.update(
                        {
                            "long_name": "zero-based time index in the original source store",
                            "comment": "Use with the fieldspacenn_source_paths attribute for exact source selection.",
                        }
                    )
                    for (variable, data_array), metadata_info in zip(
                        decoded_arrays.items(), grid_metadata
                    ):
                        source, source_array, level_dim, level_indices = metadata_info
                        piece = _attach_grid_mapping_metadata(
                            piece,
                            source,
                            source_array,
                            record_time_indices,
                            level_dim,
                            level_indices,
                        )
                    piece.attrs.update(
                        {
                            "fieldspacenn_data_kind": "decoded_reconstruction",
                            "fieldspacenn_model_profile": str(manifest["model_profile"]),
                            "fieldspacenn_config_path": str(experiment.config_path),
                            "fieldspacenn_checkpoint_path": str(experiment.checkpoint_path),
                            "fieldspacenn_source_paths": json.dumps(locations, sort_keys=True),
                            "fieldspacenn_source_global_attrs": json.dumps(
                                source_global_attrs, sort_keys=True, default=str
                            ),
                        }
                    )

                    if first_write:
                        encoding = {}
                        for variable, data_array in decoded_arrays.items():
                            chunks = tuple(
                                1
                                if dim == "time"
                                else min(int(data_array.sizes[dim]), int(spatial_chunk_size))
                                if dim in ("cell", "ncells")
                                else int(data_array.sizes[dim])
                                for dim in data_array.dims
                            )
                            encoding[variable] = {"chunks": chunks}
                        piece.to_zarr(
                            staging_path,
                            mode="w",
                            consolidated=False,
                            encoding=encoding,
                            zarr_format=2,
                        )
                        first_write = False
                    else:
                        # Static spatial/vertical coordinates already live in the store;
                        # append only decoded variables and their original time coordinate.
                        append_piece = xr.Dataset(
                            {
                                variable: xr.DataArray(
                                    data_array.data,
                                    dims=data_array.dims,
                                    attrs=copy.deepcopy(data_array.attrs),
                                )
                                for variable, data_array in decoded_arrays.items()
                            },
                            coords={
                                name: coordinate
                                for name, coordinate in piece.coords.items()
                                if "time" in coordinate.dims
                            },
                            attrs=copy.deepcopy(piece.attrs),
                        )
                        append_piece.to_zarr(
                            staging_path,
                            mode="a",
                            append_dim="time",
                            consolidated=False,
                            zarr_format=2,
                        )
                    written_records += 1
                    written_times += len(record_time_indices)
                    progress.update(1)
        finally:
            progress.close()

        if first_write:
            raise ValueError("The artifact manifest contains no decoded records.")
        if written_records != int(manifest["sample_count"]):
            raise RuntimeError(
                f"Expected {manifest['sample_count']} records, exported {written_records}."
            )
        zarr.consolidate_metadata(str(staging_path))

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        staging_path.rename(output_path)
    except Exception:
        if staging_path.exists():
            if staging_path.is_dir():
                shutil.rmtree(staging_path)
            else:
                staging_path.unlink()
        raise
    finally:
        for source in source_datasets.values():
            source.close()

    with xr.open_zarr(output_path, decode_times=False, consolidated=True) as exported:
        exported_variables = list(experiment.configured_variables)
        for variable in exported_variables:
            if variable not in exported:
                raise RuntimeError(f"Decoded Zarr is missing `{variable}`.")
        sizes = {name: int(size) for name, size in exported.sizes.items()}
        if int(exported.sizes.get("time", -1)) != written_times:
            raise RuntimeError("Decoded Zarr time dimension failed validation.")
    store_bytes = sum(
        path.stat().st_size for path in output_path.rglob("*") if path.is_file()
    )
    return {
        "path": str(output_path),
        "variables": exported_variables,
        "records": written_records,
        "time_steps": written_times,
        "sizes": sizes,
        "store_bytes": store_bytes,
    }


def load_plotting_example(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    plot_sample_index: int,
    plot_variable: str,
    plot_level_index: int,
    configured_variables: Sequence[str],
    configured_variable_groups: Mapping[str, Sequence[str]],
    dataset: Any,
    load_sample: Callable[[int], Any],
    codec: Any,
    device: torch.device,
    max_zoom: int,
) -> Dict[str, Any]:
    """Reload and decode one artifact record for plotting without rerunning evaluation."""
    from fieldspacenn.src.modules.grids.grid_utils import decode_zooms

    sample_count = int(manifest["sample_count"])
    if not 0 <= plot_sample_index < sample_count:
        raise IndexError(f"PLOT_SAMPLE_INDEX must be in [0, {sample_count - 1}].")
    if plot_variable not in configured_variables:
        raise ValueError(f"PLOT_VARIABLE must be one of {list(configured_variables)}.")
    group_names = list(configured_variable_groups)
    matching_groups = [
        index
        for index, group in enumerate(configured_variable_groups.values())
        if plot_variable in group
    ]
    if len(matching_groups) != 1:
        raise ValueError(f"Could not uniquely locate {plot_variable!r} in the variable groups.")
    group_index = matching_groups[0]
    group_variables = list(configured_variable_groups[group_names[group_index]])

    selected_record = None
    for shard_info in manifest["shards"]:
        records = load_shard(Path(output_dir) / shard_info["name"])
        selected_record = next(
            (
                record
                for record in records
                if int(record["run_position"]) == int(plot_sample_index)
            ),
            None,
        )
        if selected_record is not None:
            break
    if selected_record is None:
        raise RuntimeError(f"No artifact record found for run position {plot_sample_index}.")

    dataset_index = int(selected_record["dataset_index"])
    _, target_groups, _, _, _ = load_sample(dataset_index)
    metadata_device = to_device(selected_record["metadata"], device)
    strings_device = to_device(selected_record["strings"], device)
    with torch.inference_mode():
        decoded = codec.decompress(
            strings_device,
            metadata_device,
            sample_configs=metadata_device["sample_configs"],
            out_zoom=max_zoom,
        )

    sample_configs = selected_record["metadata"]["sample_configs"]
    reconstructed_group = decode_zooms(
        to_cpu(decoded["x_hat"][group_index]), sample_configs=sample_configs, out_zoom=max_zoom
    )
    target_group = decode_zooms(
        to_cpu(target_groups[group_index]), sample_configs=sample_configs, out_zoom=max_zoom
    )
    reconstructed = reconstructed_group[max_zoom].float()
    target = target_group[max_zoom].float()
    if reconstructed.shape != target.shape:
        raise ValueError(f"Reconstruction/target mismatch: {reconstructed.shape} vs {target.shape}")

    variable_index = group_variables.index(plot_variable)
    if reconstructed.shape[1] != len(group_variables):
        raise ValueError("Decoded variable axis does not match the configured variable list.")
    n_levels = int(reconstructed.shape[-2])
    if not 0 <= int(plot_level_index) < n_levels:
        raise IndexError(f"PLOT_LEVEL_INDEX must be in [0, {n_levels - 1}] for {plot_variable!r}.")
    normalizer_zoom = max(int(zoom) for zoom in dataset.var_normalizers.keys())
    normalizer = dataset.var_normalizers[normalizer_zoom][plot_variable]
    prediction = normalizer.denormalize(
        reconstructed[:, variable_index : variable_index + 1]
    )
    target_physical = normalizer.denormalize(target[:, variable_index : variable_index + 1])
    prediction_map = prediction[..., plot_level_index : plot_level_index + 1, :].squeeze().numpy()
    target_map = target_physical[..., plot_level_index : plot_level_index + 1, :].squeeze().numpy()
    return {
        "dataset_index": dataset_index,
        "time_index": dataset_time_index(dataset, max_zoom, dataset_index),
        "target": target_map,
        "prediction": prediction_map,
        "level_index": int(plot_level_index),
    }


def plot_rmse_vs_compression(
    target_map: np.ndarray,
    prediction_map: np.ndarray,
    model_compression_ratio: float,
    original_zoom: int,
    variable_name: str,
    output_path: Path,
    n_lower_levels: int = 3,
) -> Dict[str, Any]:
    """Compare a reconstruction with lower-resolution HEALPix baselines.

    Each baseline degrades the ground-truth map to a lower HEALPix level by
    averaging child pixels and upgrades it back in nested ordering. Upgrading
    assigns each coarse value to all of its children, which is nearest-neighbor
    interpolation on the HEALPix hierarchy. Baseline compression factors are
    theoretical value-count ratios; the model point uses its measured artifact
    compression ratio.
    """
    import healpy as hp
    import matplotlib.pyplot as plt

    target = np.asarray(target_map, dtype=np.float64).squeeze()
    prediction = np.asarray(prediction_map, dtype=np.float64).squeeze()
    original_zoom = int(original_zoom)
    n_lower_levels = int(n_lower_levels)
    model_compression_ratio = float(model_compression_ratio)

    if target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError(
            f"Expected matching one-dimensional maps, got {target.shape} and {prediction.shape}."
        )
    expected_pixels = hp.nside2npix(2**original_zoom)
    if target.size != expected_pixels:
        raise ValueError(
            f"Expected {expected_pixels} pixels for HPX{original_zoom}, got {target.size}."
        )
    if not 1 <= n_lower_levels <= original_zoom:
        raise ValueError(
            f"n_lower_levels must be between 1 and {original_zoom}, got {n_lower_levels}."
        )
    if not np.isfinite(model_compression_ratio) or model_compression_ratio <= 0:
        raise ValueError("model_compression_ratio must be finite and positive.")

    lower_zooms = list(range(original_zoom - 1, original_zoom - n_lower_levels - 1, -1))
    baseline_ratios = [4 ** (original_zoom - zoom) for zoom in lower_zooms]
    baseline_rmse = []
    for zoom in lower_zooms:
        coarse_map = hp.ud_grade(
            target,
            nside_out=2**zoom,
            order_in="NESTED",
            order_out="NESTED",
        )
        interpolated_map = hp.ud_grade(
            coarse_map,
            nside_out=2**original_zoom,
            order_in="NESTED",
            order_out="NESTED",
        )
        baseline_rmse.append(float(np.sqrt(np.mean((interpolated_map - target) ** 2))))

    model_rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        baseline_ratios,
        baseline_rmse,
        marker="o",
        linewidth=2,
        color="tab:blue",
        label="Lower-resolution baseline",
    )
    ax.scatter(
        [model_compression_ratio],
        [model_rmse],
        marker="*",
        s=180,
        color="tab:green",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
        label="Hyperprior autoencoder",
    )
    for ratio, rmse, zoom in zip(baseline_ratios, baseline_rmse, lower_zooms):
        ax.annotate(
            f"HPX{zoom}",
            (ratio, rmse),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    ax.annotate(
        f"model · {model_compression_ratio:.1f}×",
        (model_compression_ratio, model_rmse),
        xytext=(7, 7),
        textcoords="offset points",
        color="tab:green",
    )
    ax.set_xscale("log", base=4)
    ax.set_xticks(baseline_ratios)
    ax.set_xticklabels([f"{ratio}×" for ratio in baseline_ratios])
    ax.set_xlabel(f"Compression relative to HPX{original_zoom} value count")
    ax.set_ylabel(f"{variable_name} RMSE (physical units)")
    ax.set_title("Selected-sample rate–distortion comparison")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(frameon=False)
    positive_x = [*baseline_ratios, model_compression_ratio]
    ax.set_xlim(min(positive_x) / 1.5, max(positive_x) * 1.5)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()

    return {
        "original_zoom": original_zoom,
        "baseline_zooms": lower_zooms,
        "baseline_compression_ratios": baseline_ratios,
        "baseline_rmse": baseline_rmse,
        "model_compression_ratio": model_compression_ratio,
        "model_rmse": model_rmse,
        "output_path": str(output_path),
    }

@dataclass(frozen=True)
class DataInspection:
    """Validated common geometry and dimensional metadata for input variables."""

    n_timesteps: int
    n_cells: int
    zoom: int
    variables: Dict[str, Dict[str, Any]]
    variable_groups: Dict[str, list[str]]
    group_levels: Dict[str, int]


def _zarr_drop_variables(path: Any, keep_variables: Sequence[str]) -> Optional[list[str]]:
    """Identify unrelated consolidated-Zarr arrays that xarray can skip."""
    if _is_remote_location(path):
        return None
    metadata_path = Path(path) / ".zmetadata"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]
        array_names = {
            key[: -len("/.zarray")]
            for key in metadata
            if key.endswith("/.zarray") and "/" not in key[: -len("/.zarray")]
        }
        keep = set(str(variable) for variable in keep_variables)
        for variable in list(keep):
            keep.update(
                metadata.get(f"{variable}/.zattrs", {}).get("_ARRAY_DIMENSIONS", [])
            )
        return sorted(array_names - keep)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _open_selected_dataset(path: Any, variables: Sequence[str]) -> xr.Dataset:
    """Open requested variables without constructing every array in a large Zarr store."""
    location = _resolve_data_location(path)
    kwargs: Dict[str, Any] = {
        "decode_times": False,
        "create_default_indexes": False,
    }
    drop_variables = _zarr_drop_variables(location, variables)
    if drop_variables:
        kwargs["drop_variables"] = drop_variables
    return xr.open_dataset(location, **kwargs)


def validate_controls(
    max_steps: int,
    in_zooms: Sequence[int],
    variables: Mapping[str, Any],
    model_complexity: float,
    project_name: str,
    run_name: str,
    batch_size: int = 1,
) -> Tuple[Tuple[int, int, int], int, int]:
    """Validate editable controls and return sorted zooms, attention size, interval."""
    variable_groups, timestep_selection = grouped_variable_configuration(variables)
    variable_specs = {
        variable: specification
        for group in variable_groups.values()
        for variable, specification in group.items()
    }
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("MAX_STEPS must be a positive integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    if len(in_zooms) != 3 or len(set(in_zooms)) != 3:
        raise ValueError("IN_ZOOMS must contain exactly three distinct zoom levels.")
    if any(not isinstance(zoom, int) or isinstance(zoom, bool) or zoom < 0 for zoom in in_zooms):
        raise ValueError("Every input zoom must be a non-negative integer.")
    if not variable_specs:
        raise ValueError("VARIABLES must contain at least one variable.")
    if timestep_selection is not None:
        expand_timestep_selection(timestep_selection)
    for variable, specification in variable_specs.items():
        if not str(variable).strip():
            raise ValueError("Variable names must be non-empty strings.")
        if not isinstance(specification, Mapping) or not specification.get("path"):
            raise ValueError(f"Variable `{variable}` needs a non-empty `path`.")
        _normalize_level_indices(specification.get("level_indices"), variable)
    if not np.isfinite(model_complexity) or model_complexity <= 0:
        raise ValueError("MODEL_COMPLEXITY must be finite and positive.")
    attention_dim = 256 * float(model_complexity)
    if not attention_dim.is_integer():
        raise ValueError("256 * MODEL_COMPLEXITY must be an integer.")
    attention_dim = int(attention_dim)
    if attention_dim % 32:
        raise ValueError(
            "256 * MODEL_COMPLEXITY must be divisible by n_head_channels=32."
        )
    if not str(project_name).strip() or not str(run_name).strip():
        raise ValueError("PROJECT_NAME and RUN_NAME must be non-empty strings.")
    zooms = tuple(sorted(int(zoom) for zoom in in_zooms))
    return zooms, attention_dim, max(1, round(max_steps / 4))


def _spatial_dimension(array: xr.DataArray) -> str:
    candidates = [dim for dim in array.dims if dim in ("cell", "ncells")]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one `cell` or `ncells` dimension, got {array.dims}."
        )
    return candidates[0]


def _zoom_from_cells(n_cells: int) -> int:
    if n_cells <= 0 or n_cells % 12:
        raise ValueError(f"{n_cells} cells do not define a complete HEALPix grid.")
    nside_squared = n_cells // 12
    nside = math.isqrt(nside_squared)
    if nside * nside != nside_squared or nside & (nside - 1):
        raise ValueError(f"{n_cells} cells do not define a power-of-two HEALPix grid.")
    return int(math.log2(nside))


def inspect_variable_stores(
    variables: Mapping[str, Any],
    expected_zoom: int,
    minimum_timesteps: int = 21,
) -> DataInspection:
    """Validate grouped dimensions, level selections, resolution, and alignment."""
    variable_groups, timestep_selection = grouped_variable_configuration(variables)
    variable_specs = {
        variable: specification
        for group in variable_groups.values()
        for variable, specification in group.items()
    }
    if not variable_specs:
        raise ValueError("VARIABLES must contain at least one variable.")
    unique_paths = {
        _resolve_data_location(specification["path"])
        for specification in variable_specs.values()
    }
    compare_coordinates = len(unique_paths) > 1
    reference_time = None
    reference_cells = None
    reference_n_time = None
    reference_n_cells = None
    details: Dict[str, Dict[str, Any]] = {}
    group_levels: Dict[str, int] = {}

    for group_name, variable, specification in (
        (group_name, variable, specification)
        for group_name, group_variables in variable_groups.items()
        for variable, specification in group_variables.items()
    ):
        path = _resolve_data_location(specification["path"])
        if not _is_remote_location(path) and not Path(path).exists():
            raise FileNotFoundError(f"Input store for `{variable}` does not exist: {path}")
        with _open_selected_dataset(path, [variable]) as dataset:
            if variable not in dataset:
                raise KeyError(f"Variable `{variable}` is missing from `{path}`.")
            array = dataset[variable]
            if "time" not in array.dims:
                raise ValueError(f"Variable `{variable}` has no time dimension: {array.dims}")
            spatial_dim = _spatial_dimension(array)
            other_dims = [dim for dim in array.dims if dim not in ("time", spatial_dim)]
            level_indices = _normalize_level_indices(
                specification.get("level_indices"), variable
            )
            if not other_dims and level_indices is not None:
                raise ValueError(
                    f"Variable `{variable}` is 2-D in time/space and must use level_indices=None."
                )
            if len(other_dims) > 1:
                raise ValueError(
                    f"Variable `{variable}` has unsupported extra dimensions {other_dims}; "
                    "field variables support at most one depth dimension."
                )
            if other_dims and level_indices is not None:
                level_size = int(array.sizes[other_dims[0]])
                invalid = [
                    index for index in level_indices if not -level_size <= index < level_size
                ]
                if invalid:
                    raise IndexError(
                        f"level_indices={invalid} is outside `{other_dims[0]}` size {level_size} "
                        f"for `{variable}`."
                    )

            if group_name == "2D":
                selected_levels = (
                    1 if not other_dims else (
                        int(array.sizes[other_dims[0]])
                        if level_indices is None
                        else len(level_indices)
                    )
                )
                if selected_levels != 1:
                    raise ValueError(
                        f"2D variable `{variable}` must resolve to exactly one level; "
                        f"got {selected_levels}."
                    )
            else:
                if not other_dims:
                    raise ValueError(
                        f"3D variable `{variable}` needs one vertical dimension in addition "
                        f"to time and {spatial_dim}."
                    )
                selected_levels = (
                    int(array.sizes[other_dims[0]])
                    if level_indices is None
                    else len(level_indices)
                )
            previous_levels = group_levels.setdefault(group_name, selected_levels)
            if previous_levels != selected_levels:
                raise ValueError(
                    f"All variables in group {group_name!r} must use the same number of levels; "
                    f"`{variable}` has {selected_levels}, expected {previous_levels}."
                )

            n_time = int(array.sizes["time"])
            n_cells = int(array.sizes[spatial_dim])
            zoom = _zoom_from_cells(n_cells)
            if zoom != int(expected_zoom):
                raise ValueError(
                    f"Variable `{variable}` is HPX{zoom} ({n_cells} cells), but the highest "
                    f"configured input zoom is HPX{expected_zoom}."
                )
            time_values = (
                np.asarray(dataset["time"].values)
                if compare_coordinates and "time" in dataset.coords
                else None
            )
            cell_values = (
                np.asarray(dataset[spatial_dim].values)
                if compare_coordinates and spatial_dim in dataset.coords
                else None
            )
            if reference_n_time is None:
                reference_n_time, reference_n_cells = n_time, n_cells
                reference_time, reference_cells = time_values, cell_values
            else:
                if (n_time, n_cells) != (reference_n_time, reference_n_cells):
                    raise ValueError(
                        f"Store for `{variable}` has shape (time={n_time}, cells={n_cells}); "
                        f"expected ({reference_n_time}, {reference_n_cells})."
                    )
                if reference_time is not None and time_values is not None and not np.array_equal(
                    reference_time, time_values
                ):
                    raise ValueError(f"Time coordinates for `{variable}` are not aligned.")
                if reference_cells is not None and cell_values is not None and not np.array_equal(
                    reference_cells, cell_values
                ):
                    raise ValueError(f"Cell coordinates for `{variable}` are not aligned.")

            details[str(variable)] = {
                "path": str(path),
                "dims": list(array.dims),
                "shape": [int(size) for size in array.shape],
                "spatial_dim": spatial_dim,
                "level_dim": other_dims[0] if other_dims else None,
                "level_indices": level_indices,
                "group": group_name,
                "selected_levels": selected_levels,
            }

    assert reference_n_time is not None and reference_n_cells is not None
    selected_count = reference_n_time
    if timestep_selection is not None:
        selected_count = len(
            expand_timestep_selection(timestep_selection, n_timesteps=reference_n_time)
        )
    if selected_count < int(minimum_timesteps):
        raise ValueError(
            f"At least {minimum_timesteps} timesteps are required; "
            f"found {selected_count}."
        )
    return DataInspection(
        n_timesteps=reference_n_time,
        n_cells=reference_n_cells,
        zoom=int(expected_zoom),
        variables=details,
        variable_groups={name: list(group) for name, group in variable_groups.items()},
        group_levels=group_levels,
    )


def _stream_array_statistics(
    array: xr.DataArray,
    *,
    train_stop: int,
    selected_timesteps: Optional[Sequence[int]],
    time_chunk: int,
) -> Tuple[int, float, float]:
    """Return finite count, mean, and population standard deviation."""
    count = 0
    mean = 0.0
    m2 = 0.0
    n_selected = int(train_stop) if selected_timesteps is None else len(selected_timesteps)
    for start in range(0, n_selected, int(time_chunk)):
        selector = (
            slice(start, min(start + time_chunk, int(train_stop)))
            if selected_timesteps is None
            else selected_timesteps[start : start + time_chunk]
        )
        values = np.asarray(array.isel(time=selector).values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        chunk_count = int(values.size)
        chunk_mean = float(values.mean(dtype=np.float64))
        chunk_m2 = float(np.square(values - chunk_mean).sum(dtype=np.float64))
        if count == 0:
            count, mean, m2 = chunk_count, chunk_mean, chunk_m2
        else:
            delta = chunk_mean - mean
            total = count + chunk_count
            mean += delta * chunk_count / total
            m2 += chunk_m2 + delta * delta * count * chunk_count / total
            count = total
    std = math.sqrt(m2 / count) if count else float("nan")
    return count, mean, std


def calculate_normalization(
    variables: Mapping[str, Any],
    train_stop: int,
    output_path: Path,
    time_chunk: int = 16,
    timesteps: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Stream exact finite-value population mean/std statistics and write JSON."""
    if train_stop <= 0:
        raise ValueError("The training split must contain at least one timestep.")
    if time_chunk <= 0:
        raise ValueError("time_chunk must be positive.")
    variable_groups, _ = grouped_variable_configuration(variables)
    selected_timesteps = None if timesteps is None else [int(index) for index in timesteps]
    if selected_timesteps is not None and not selected_timesteps:
        raise ValueError("Normalization requires at least one training timestep.")
    norm: Dict[str, Any] = {}
    for group_name, variable, specification in (
        (group_name, variable, specification)
        for group_name, group_variables in variable_groups.items()
        for variable, specification in group_variables.items()
    ):
        with _open_selected_dataset(
            specification["path"], [variable]
        ) as dataset:
            array = dataset[variable]
            spatial_dim = _spatial_dimension(array)
            level_dims = [dim for dim in array.dims if dim not in ("time", spatial_dim)]
            level_indices = _normalize_level_indices(
                specification.get("level_indices"), variable
            )
            if group_name == "2D":
                if level_dims:
                    source_level = (
                        level_indices[0]
                        if level_indices is not None
                        else 0
                    )
                    array = array.isel({level_dims[0]: source_level})
                count, mean, std = _stream_array_statistics(
                    array.transpose("time", spatial_dim),
                    train_stop=train_stop,
                    selected_timesteps=selected_timesteps,
                    time_chunk=time_chunk,
                )
            else:
                # Store full-source per-level statistics. The loader applies
                # level_indices to these arrays before normalizing selected data.
                level_dim = level_dims[0]
                counts: list[int] = []
                means: list[float] = []
                stds: list[float] = []
                for level in range(int(array.sizes[level_dim])):
                    count, mean, std = _stream_array_statistics(
                        array.isel({level_dim: level}).transpose("time", spatial_dim),
                        train_stop=train_stop,
                        selected_timesteps=selected_timesteps,
                        time_chunk=time_chunk,
                    )
                    counts.append(count)
                    means.append(mean)
                    stds.append(std)
                count, mean, std = counts, means, stds

        counts_to_check = [count] if isinstance(count, int) else count
        means_to_check = np.atleast_1d(mean).astype(float)
        stds_to_check = np.atleast_1d(std).astype(float)
        if any(value <= 0 for value in counts_to_check):
            raise ValueError(f"Variable `{variable}` has an empty training level.")
        if not np.isfinite(means_to_check).all() or not np.isfinite(stds_to_check).all() or np.any(stds_to_check <= 0):
            raise ValueError(
                f"Invalid normalization for `{variable}`: mean={mean}, std={std}."
            )
        norm[str(variable)] = {
            "normalizer": {"class": "MeanStdNormalizer"},
            "stats": {"mean": mean, "std": std},
            "finite_count": count,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
    return norm


def _remap_zoom_list(values: Sequence[int], zoom_map: Mapping[int, int]) -> list[int]:
    return [int(zoom_map.get(int(value), int(value))) for value in values]


def _replace_zoom_keys(mapping: Mapping[Any, Any], zoom_map: Mapping[int, int]) -> Dict[int, Any]:
    return {int(zoom_map.get(int(key), int(key))): copy.deepcopy(value) for key, value in mapping.items()}


def _configure_attention_blocks(
    blocks: Mapping[str, Any],
    zoom_map: Mapping[int, int],
    attention_dim: int,
    token_zoom: int,
    token_len_depth: Sequence[int],
) -> None:
    for block in blocks.values():
        if "att_dim" not in block:
            continue
        for field in ("q_zooms", "kv_zooms", "target_zooms", "in_zooms", "out_zooms"):
            if field in block:
                block[field] = _remap_zoom_list(block[field], zoom_map)
        block.att_dim = int(attention_dim)
        block.token_zoom = int(token_zoom)
        block.token_len_depth = [int(length) for length in token_len_depth]
        if "rank_depth" in block:
            block.rank_depth = [None] * len(token_len_depth)
        if block.get("embed_confs") is not None:
            block.embed_confs.input_zoom = int(token_zoom)


def build_pretraining_config(
    base_config_path: Path,
    *,
    max_steps: int,
    in_zooms: Sequence[int],
    variables: Mapping[str, Any],
    model_complexity: float,
    project_name: str,
    run_name: str,
    run_dir: Path,
    norm_path: Path,
    n_timesteps: int,
    variable_group_levels: Optional[Mapping[str, int]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DictConfig:
    """Create a fully remapped, self-contained pretraining configuration."""
    variable_groups, timestep_selection = grouped_variable_configuration(variables)
    variable_specs = {
        variable: specification
        for group in variable_groups.values()
        for variable, specification in group.items()
    }
    group_levels = {str(group): int(levels) for group, levels in (variable_group_levels or {}).items()}
    if "3D" in variable_groups and "3D" not in group_levels:
        explicit_counts = {
            len(indices)
            for variable, specification in variable_groups["3D"].items()
            if (indices := _normalize_level_indices(specification.get("level_indices"), variable))
            is not None
        }
        if len(explicit_counts) != 1 or any(
            specification.get("level_indices") is None
            for specification in variable_groups["3D"].values()
        ):
            raise ValueError(
                "variable_group_levels['3D'] is required when any 3D variable uses all levels."
            )
        group_levels["3D"] = explicit_counts.pop()
    group_levels.setdefault("2D", 1)
    group_names = list(variable_groups)
    group_variable_counts = [len(variable_groups[group]) for group in group_names]
    token_len_depth = [group_levels[group] for group in group_names]
    (low, medium, high), attention_dim, interval = validate_controls(
        max_steps,
        in_zooms,
        variables,
        model_complexity,
        project_name,
        run_name,
        batch_size,
    )
    timestep_splits = training_timestep_splits(variables, n_timesteps)
    cfg = OmegaConf.load(Path(base_config_path))
    zoom_map = {5: low, 7: medium, 9: high}

    cfg.project_name = str(project_name)
    cfg.run_name = str(run_name)
    cfg.run_dir = str(Path(run_dir).resolve())
    cfg.ckpt_path = None
    cfg.ckpt_path_pretrained = None

    cfg.trainer.max_steps = int(max_steps)
    cfg.trainer.accelerator = "auto"
    cfg.trainer.devices = 1
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.val_check_interval = interval
    cfg.trainer.log_every_n_steps = 1
    cfg.trainer.default_root_dir = str(
        Path(run_dir).resolve() / "snapshots" / project_name / run_name
    )
    checkpoint = cfg.trainer.callbacks[0]
    checkpoint.dirpath = cfg.trainer.default_root_dir
    checkpoint.every_n_train_steps = interval

    cfg.logger.save_dir = cfg.trainer.default_root_dir
    cfg.logger.name = "csv_logs"
    cfg.logger.version = ""

    cfg.model.stage = "pretrain"
    cfg.model.freeze_analysis_encoder = False
    cfg.model.freeze_synthesis_decoder = False
    loss_zooms = cfg.model.loss_config.reconstruction_loss_config.zooms
    cfg.model.loss_config.reconstruction_loss_config.zooms = _replace_zoom_keys(
        loss_zooms, zoom_map
    )

    codec = cfg.model.model
    codec.mgrids.zoom_max = high
    codec.in_zooms = [high, medium, low]
    codec.passthrough_zooms = [low]
    codec.n_groups_variables = group_variable_counts
    codec.n_groups_depths = [1] * len(group_names)
    _configure_attention_blocks(
        codec.analysis_block_configs, zoom_map, attention_dim, min(2, low), token_len_depth
    )
    _configure_attention_blocks(
        codec.synthesis_block_configs, zoom_map, attention_dim, min(2, low), token_len_depth
    )
    _configure_attention_blocks(
        codec.hyper_analysis_block_configs, zoom_map, attention_dim, min(4, medium), token_len_depth
    )
    _configure_attention_blocks(
        codec.hyper_synthesis_block_configs, zoom_map, attention_dim, min(4, medium), token_len_depth
    )

    for name, block in codec.analysis_block_configs.items():
        if "down" in str(name).lower() or ("in_zooms" in block and "field_zoom" in block):
            block.in_zooms = [medium, high]
            block.target_zooms = [medium]
            block.field_zoom = medium
            block.out_zooms = [low, medium]
    for name, block in codec.synthesis_block_configs.items():
        if "up" in str(name).lower() or ("in_zooms" in block and "field_zoom" in block):
            block.in_zooms = [medium]
            block.target_zooms = [high]
            block.field_zoom = min(4, medium)
            block.out_zooms = [low, medium, high]

    cfg.mgrids.zoom_max = high
    cfg.embedding.MGEmbedder.n_variables = len(variable_specs)
    cfg.dataloader.dataset.norm_dict = str(Path(norm_path).resolve())
    cfg.dataloader.dataset.sampling_zooms = _replace_zoom_keys(
        cfg.dataloader.dataset.sampling_zooms, zoom_map
    )
    cfg.dataloader.datamodule.batch_size = int(batch_size)
    cfg.dataloader.datamodule.num_workers = int(num_workers)

    anchor = _resolve_data_location(next(iter(variable_specs.values()))["path"])
    zoom_files = {zoom: {"files": [anchor]} for zoom in (low, medium, high)}
    variable_files = {
        str(variable): {
            "files": [_resolve_data_location(specification["path"])]
        }
        for variable, specification in variable_specs.items()
    }
    variable_group = {
        group_name: {
            str(variable): (
                {}
                if specification.get("level_indices") is None
                else {
                    "level_indices": (
                        _normalize_level_indices(specification["level_indices"], variable)
                        if group_name == "3D"
                        else _normalize_level_indices(specification["level_indices"], variable)[0]
                    )
                }
            )
            for variable, specification in group.items()
        }
        for group_name, group in variable_groups.items()
    }
    cfg.data_split = {}
    for split, selected_timesteps in timestep_splits.items():
        if timestep_selection is None:
            start, stop = selected_timesteps[0], selected_timesteps[-1] + 1
            configured_timesteps: list[Any] = [f"{start}-{stop}"]
        else:
            configured_timesteps = list(selected_timesteps)
        cfg.data_split[split] = {
            "source": copy.deepcopy(zoom_files),
            "target": copy.deepcopy(zoom_files),
            "variable_files": copy.deepcopy(variable_files),
            "timesteps": configured_timesteps,
            "variables": copy.deepcopy(variable_group),
        }
    cfg.data_zooms = copy.deepcopy(zoom_files)
    cfg.data_variables = copy.deepcopy(variable_group)
    return cfg


def build_finetuning_config(pretraining_config: DictConfig, checkpoint_path: Path) -> DictConfig:
    """Deep-copy pretraining settings and apply the joint fine-tuning recipe."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretraining checkpoint does not exist: {checkpoint_path}")
    cfg = copy.deepcopy(pretraining_config)
    cfg.run_name = f"{pretraining_config.run_name}_finetune"
    cfg.model.stage = "joint"
    cfg.model.freeze_analysis_encoder = True
    cfg.model.freeze_synthesis_decoder = False
    cfg.ckpt_path = None
    cfg.ckpt_path_pretrained = str(checkpoint_path)
    root = Path(str(cfg.run_dir)).resolve()
    cfg.trainer.default_root_dir = str(root / "snapshots" / cfg.project_name / cfg.run_name)
    cfg.trainer.callbacks[0].dirpath = cfg.trainer.default_root_dir
    cfg.logger.save_dir = cfg.trainer.default_root_dir
    cfg.logger.name = "csv_logs"
    return cfg


def validation_schedule(max_steps: int) -> Dict[str, Any]:
    """Return the rounded interval and expected validation step positions."""
    interval = max(1, round(int(max_steps) / 4))
    steps = list(range(interval, int(max_steps) + 1, interval))
    exact_four = len(steps) == 4 and steps[-1] == int(max_steps)
    if not exact_four:
        warnings.warn(
            f"Rounded interval {interval} yields validation at {steps}; this is not exactly "
            "four evenly spaced runs ending at MAX_STEPS.",
            stacklevel=2,
        )
    return {"interval": interval, "validation_steps": steps, "exact_four": exact_four}


def prepare_run_directory(path: Path, overwrite: bool) -> Path:
    """Fail safely or remove only artifacts owned by this notebook from a run directory."""
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {path}. Choose another RUN_NAME or set "
            "OVERWRITE_EXISTING_RUN=True."
        )
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        known_files = {
            "composed_config.yaml",
            "last.ckpt",
            "training_loss.png",
            "input_zoom_preview.png",
            "input_zoom_decomposition.png",
            "normalization_mean_std.json",
        }
        for child in list(path.iterdir()):
            if child.is_file() and (child.name in known_files or child.suffix == ".ckpt"):
                child.unlink()
            elif child.is_dir() and child.name in {"csv_logs", "lightning_logs"}:
                shutil.rmtree(child)
    return path


def save_config(config: DictConfig, path: Path) -> Path:
    """Resolve interpolations and save the exact configuration used by a stage."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=path, resolve=True)
    return path


def _sampled_variable_names(embedding_group: Mapping[str, Any]) -> list[str]:
    """Normalize collated variable-name metadata for one data group."""
    names = []
    for name in embedding_group["variable_names_sampled"]:
        if isinstance(name, (list, tuple)) and len(name) == 1:
            name = name[0]
        names.append(str(name))
    return names


def plot_input_zooms(
    dataset: Any,
    output_path: Optional[Path] = None,
    sample_index: int = 0,
) -> Tuple[Any, Dict[str, Dict[int, np.ndarray]]]:
    """Plot a composed, normalized dataset sample at every configured input zoom."""
    import healpy as hp
    import matplotlib.pyplot as plt
    from fieldspacenn.src.data.pl_data_module import BatchReshapeAllocator
    from fieldspacenn.src.modules.grids.grid_utils import decode_zooms
    from fieldspacenn.src.utils.helpers import merge_sampling_dicts

    collator = BatchReshapeAllocator(dataset)
    sources, _, _, embeddings, patch_indices = collator([dataset[int(sample_index)]])
    sampling = dataset.sampling_zooms_collate or dataset.sampling_zooms
    if OmegaConf.is_config(sampling):
        sampling = OmegaConf.to_container(sampling, resolve=True)
    sampling = {int(zoom): values for zoom, values in sampling.items()}
    sampling = merge_sampling_dicts(sampling, patch_indices)
    zooms = sorted({int(zoom) for group in sources if group for zoom in group})
    maps: Dict[str, Dict[int, np.ndarray]] = {}
    for group_index, source_group in enumerate(sources):
        if not source_group:
            continue
        variable_names = _sampled_variable_names(embeddings[group_index])
        for zoom in zooms:
            composed = decode_zooms(source_group, sampling, zoom)[zoom]
            if composed.shape[1] != len(variable_names):
                raise ValueError("Variable names do not match the composed group tensor.")
            for variable_index, variable in enumerate(variable_names):
                for level_index in range(composed.shape[-2]):
                    label = (
                        variable
                        if composed.shape[-2] == 1
                        else f"{variable} [level {level_index}]"
                    )
                    values = composed[
                        :, variable_index : variable_index + 1, :, :, level_index : level_index + 1
                    ]
                    maps.setdefault(label, {})[zoom] = values.detach().cpu().numpy().squeeze()
    variable_names = list(maps)

    # Healpy creates its own projected axes. Pre-creating regular matplotlib
    # subplots causes the Mollweide axes to overlap instead of occupying the
    # requested grid positions.
    fig = plt.figure(figsize=(4.3 * len(zooms), 3.2 * len(variable_names)))
    for row, variable in enumerate(variable_names):
        finite = np.concatenate(
            [values[np.isfinite(values)] for values in maps[variable].values()]
        )
        vmin, vmax = np.quantile(finite, [0.01, 0.99])
        if vmin == vmax:
            vmin, vmax = float(finite.min()), float(finite.max() + 1e-12)
        for column, zoom in enumerate(zooms):
            hp.mollview(
                maps[variable][zoom],
                nest=True,
                fig=fig.number,
                sub=(len(variable_names), len(zooms), row * len(zooms) + column + 1),
                title=f"{variable} · HPX{zoom}",
                min=vmin,
                max=vmax,
                cmap="RdBu_r",
                hold=False,
            )
    fig.suptitle("Raw data before decomposition at different resolutions", y=0.99)
    fig.subplots_adjust(top=0.91, bottom=0.05, hspace=0.32, wspace=0.12)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, maps


def plot_decomposed_input_zooms(
    dataset: Any,
    output_path: Optional[Path] = None,
    sample_index: int = 0,
) -> Tuple[Any, Dict[str, Dict[int, np.ndarray]]]:
    """Plot the normalized base field and residual tensors supplied to the model."""
    import healpy as hp
    import matplotlib.pyplot as plt
    from fieldspacenn.src.data.pl_data_module import BatchReshapeAllocator

    collator = BatchReshapeAllocator(dataset)
    sources, _, _, embeddings, _ = collator([dataset[int(sample_index)]])
    zooms = sorted({int(zoom) for group in sources if group for zoom in group})
    lowest_zoom = min(zooms)
    maps: Dict[str, Dict[int, np.ndarray]] = {}
    for group_index, source_group in enumerate(sources):
        if not source_group:
            continue
        variable_names_group = _sampled_variable_names(embeddings[group_index])
        for zoom in zooms:
            component = source_group[zoom]
            if component.shape[1] != len(variable_names_group):
                raise ValueError("Variable names do not match the decomposed group tensor.")
            for variable_index, variable in enumerate(variable_names_group):
                for level_index in range(component.shape[-2]):
                    label = (
                        variable
                        if component.shape[-2] == 1
                        else f"{variable} [level {level_index}]"
                    )
                    values = component[
                        :, variable_index : variable_index + 1, :, :, level_index : level_index + 1
                    ]
                    maps.setdefault(label, {})[zoom] = values.detach().cpu().numpy().squeeze()
    variable_names = list(maps)

    fig = plt.figure(figsize=(4.3 * len(zooms), 3.2 * len(variable_names)))
    for row, variable in enumerate(variable_names):
        base_values = maps[variable][lowest_zoom]
        base_finite = base_values[np.isfinite(base_values)]
        base_min, base_max = np.quantile(base_finite, [0.01, 0.99])

        residual_values = np.concatenate(
            [
                maps[variable][zoom][np.isfinite(maps[variable][zoom])]
                for zoom in zooms
                if zoom != lowest_zoom
            ]
        )
        residual_limit = float(np.quantile(np.abs(residual_values), 0.99))
        residual_limit = max(residual_limit, np.finfo(np.float32).eps)

        for column, zoom in enumerate(zooms):
            is_base = zoom == lowest_zoom
            hp.mollview(
                maps[variable][zoom],
                nest=True,
                fig=fig.number,
                sub=(len(variable_names), len(zooms), row * len(zooms) + column + 1),
                title=(
                    f"{variable}\nHPX{zoom} base"
                    if is_base
                    else f"{variable}\nHPX{zoom} residual"
                ),
                min=float(base_min) if is_base else -residual_limit,
                max=float(base_max) if is_base else residual_limit,
                cmap="RdBu_r",
                hold=False,
            )
    fig.suptitle("Decomposed data: base field and residuals", y=0.99)
    fig.subplots_adjust(top=0.91, bottom=0.05, hspace=0.32, wspace=0.12)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, maps


def _training_only_progress_bar() -> Any:
    """Keep Lightning's training bar while suppressing noisy validation bars."""
    from lightning.pytorch.callbacks import TQDMProgressBar
    from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

    class TrainingOnlyTQDMProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self) -> Any:
            return Tqdm(disable=True)

    return TrainingOnlyTQDMProgressBar()


def run_training_stage(
    config: DictConfig,
    train_dataset: Any,
    validation_dataset: Any,
) -> Dict[str, Any]:
    """Instantiate the model/Lightning objects and train with provided datasets."""
    from fieldspacenn.src.data.pl_data_module import DataModule
    from fieldspacenn.src.utils.helpers import load_pretrained_checkpoints

    logger = instantiate(config.logger)
    model = instantiate(config.model)
    if config.get("ckpt_path_pretrained") is not None:
        model, _ = load_pretrained_checkpoints(
            model,
            config.ckpt_path_pretrained,
            freeze_pretrained=False,
            print_keys=False,
        )
    callbacks = [instantiate(callback) for callback in config.trainer.callbacks]
    callbacks.append(_training_only_progress_bar())
    trainer = instantiate(config.trainer, logger=logger, callbacks=callbacks)
    data_module: DataModule = instantiate(
        config.dataloader.datamodule,
        dataset_train=train_dataset,
        dataset_val=validation_dataset,
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=config.get("ckpt_path"))
    checkpoint = Path(str(config.trainer.default_root_dir)) / "last.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Lightning did not create the expected checkpoint: {checkpoint}")
    return {
        "trainer": trainer,
        "model": model,
        "logger": logger,
        "log_dir": Path(logger.log_dir),
        "checkpoint": checkpoint.resolve(),
        "config": Path(str(config.trainer.default_root_dir)).resolve() / "composed_config.yaml",
    }


def load_loss_metrics(log_dir: Path) -> pd.DataFrame:
    """Load local Lightning CSV metrics and normalize loss column names."""
    metrics_path = Path(log_dir) / "metrics.csv"
    if not metrics_path.is_file():
        candidates = list(Path(log_dir).rglob("metrics.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not identify metrics.csv below {log_dir}.")
        metrics_path = candidates[0]
    frame = pd.read_csv(metrics_path)
    metric_candidates = {
        "train_loss": ["train/total_loss", "train_total_loss", "train_loss"],
        "val_loss": ["val/total_loss", "val_total_loss", "val_loss"],
        "train_compression_ratio": [
            "train/estimated_compression_ratio",
            "train_estimated_compression_ratio",
            "train_compression_ratio",
        ],
        "val_compression_ratio": [
            "val/estimated_compression_ratio",
            "val_estimated_compression_ratio",
            "val_compression_ratio",
        ],
    }
    rename = {}
    for output, candidates in metric_candidates.items():
        found = next((column for column in candidates if column in frame), None)
        if found is not None:
            rename[found] = output
    frame = frame.rename(columns=rename)
    if "step" not in frame:
        raise ValueError(f"Metrics file has no `step` column: {metrics_path}")
    if "train_loss" not in frame and "val_loss" not in frame:
        raise ValueError(f"No total-loss columns found in {metrics_path}.")
    frame.attrs["path"] = str(metrics_path)
    return frame


def plot_training_losses(
    pretraining_log_dir: Path,
    finetuning_log_dir: Path,
    output_path: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """Plot log-scale losses, fine-tuning compression ratio, and loss summaries."""
    import matplotlib.pyplot as plt

    stages = {
        "Pretraining": load_loss_metrics(pretraining_log_dir),
        "Fine-tuning": load_loss_metrics(finetuning_log_dir),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), sharey=False)
    summary: Dict[str, Any] = {}
    for axis, (name, frame) in zip(axes[:2], stages.items()):
        for column, label, color in (
            ("train_loss", "Training", "tab:blue"),
            ("val_loss", "Validation", "tab:orange"),
        ):
            if column not in frame:
                continue
            values = frame[["step", column]].dropna().sort_values("step")
            values = values[np.isfinite(values[column]) & (values[column] > 0)]
            axis.plot(values["step"], values[column], marker="o", ms=3, label=label, color=color)
        axis.set_title(name)
        axis.set_xlabel("Global step")
        axis.set_ylabel("Total loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend()
        validation = frame[["step", "val_loss"]].dropna() if "val_loss" in frame else pd.DataFrame()
        summary[name.lower().replace("-", "_")] = {
            "metrics_path": frame.attrs["path"],
            "final_validation_loss": (
                float(validation.iloc[-1]["val_loss"]) if not validation.empty else None
            ),
            "best_validation_loss": (
                float(validation["val_loss"].min()) if not validation.empty else None
            ),
        }

    finetuning = stages["Fine-tuning"]
    compression_axis = axes[2]
    for column, label, color in (
        ("train_compression_ratio", "Training", "tab:blue"),
        ("val_compression_ratio", "Validation", "tab:orange"),
    ):
        if column not in finetuning:
            continue
        values = finetuning[["step", column]].dropna().sort_values("step")
        values = values[np.isfinite(values[column]) & (values[column] > 0)]
        compression_axis.plot(
            values["step"], values[column], marker="o", ms=3, label=label, color=color
        )
    compression_axis.set_title("Fine-tuning compression ratio")
    compression_axis.set_xlabel("Global step")
    compression_axis.set_ylabel("Estimated compression ratio (×)")
    compression_axis.grid(alpha=0.25, linestyle="--")
    if compression_axis.lines:
        compression_axis.legend()
    else:
        compression_axis.text(
            0.5,
            0.5,
            "No compression-ratio metric found",
            transform=compression_axis.transAxes,
            ha="center",
            va="center",
        )
    fig.suptitle("Two-stage hyperprior optimization")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, summary
