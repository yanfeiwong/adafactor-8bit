# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

"""Model-aware parameter grouping for Adafactor8Bit.

Presets choose update settings; grouping decides which parameter roles can be
configured independently. The builder returns ordinary PyTorch parameter groups.

Grouping profiles:
    auto: Uses the preset's default topology (currently G1), with preset-specific
        quantization defaults. Vectors/norms/biases stay unquantized; embedding
        and LM-head state precision depends on the preset.
    G0 / bare: One all group for ordinary presets; apollo and fallback groups
        for APOLLO. No role-specific weight-decay or quantization protection.
    G1 / protected: Separate one_d, embedding, lm_head and other groups, plus
        apollo when selected. The first three default to no quantization or
        weight decay. Unlike auto, explicit G1 keeps this protection for every
        preset.
    G2 / topology: Separate one_d, embedding, lm_head, matrix and nd groups,
        plus apollo when selected. Embeddings use full-rank, momentum-free
        updates; LM heads use full-rank updates. one_d and embedding states
        default to unquantized. nd_factored optionally enables factorization
        for the Adafactor/CAME N-D route. Lower learning rates require explicit
        overrides; they are not inferred.

Classification and routing:
    Collect parameter aliases and ownership, classify vectors/norms/biases,
    token embeddings, LM heads, other matrices and N-D tensors, then select
    eligible APOLLO targets by name and shape. Positional/time embeddings are
    excluded from the token-embedding name heuristics. Shared parameters are
    assigned once; tied token-embedding/LM-head weights use the embedding role.
    Only non-empty groups are returned, in a stable role order.

Configuration:
    Route options override the grouping defaults. The optional dictionary
    policy overlays generated groups, then policy defaults, mapped presets and
    mapped options in that order. Group values override optimizer constructor
    defaults; omitted values inherit them. Quantization still respects each
    parameter's min_8bit_size threshold. Building groups does not cast weights
    or load optimizer state.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from torch import nn

from .presets import PRESETS, PresetSpec, _preset_names

__all__ = ["build_param_groups"]


_GROUPING_ALIASES = {
    "G0": "G0", "g0": "G0", "bare": "G0",
    "G1": "G1", "g1": "G1", "protected": "G1",
    "G2": "G2", "g2": "G2", "topology": "G2",
}

# These are final generated route names, not abstract protection labels.
_ROUTE_NAMES = {
    "all", "fallback", "other",
    "one_d", "embedding", "lm_head", "apollo", "matrix", "nd",
}

# Longest overlapping suffixes must come first (m_block_size before block_size).
_ROUTE_OPTION_SUFFIXES = (
    "m_quant_type", "v_quant_type", "c_quant_type",
    "stochastic_round_writeback",
    "min_8bit_size", "m_block_size", "block_size", "quantize", "lr",
)

_M_TYPES = {"fp32", "uf4", "uf8", "d4", "d8"}
_VC_TYPES = {"fp32", "al8", "al16"}

# Generic semantic overrides apply to both a primary path and its fallback.
_COMMON_OVERRIDE_KEYS = {
    "beta1", "beta2", "beta2_decay", "bias_correction",
    "relative_step", "scale_parameter", "scale_weight_decay", "d", "eps",
    "fira_margin", "lag_norm", "momentum_only_1d", "enable_fira_for_adafactor",
    "stochastic_round_writeback",
}

# Confidence/APOLLO controls belong only to the selected primary path. This
# keeps APOLLO's Adam-style fallback from accidentally becoming CAME.
_PRIMARY_OVERRIDE_KEYS = {
    "beta3", "eps_came",
    "apollo_factorize", "apollo_update_proj_gap", "apollo_scale_type",
    "apollo_scale", "apollo_scale_front", "apollo_eps", "apollo_cache_proj",
    "enable_fira_for_apollo",
}


def _norm_types() -> Tuple[type, ...]:
    names = (
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm",
        "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
        "LayerNorm", "GroupNorm", "LocalResponseNorm", "RMSNorm",
    )
    return tuple(getattr(nn, name) for name in names if hasattr(nn, name))


_NORM_TYPES = _norm_types()


def _named_modules_all(model: nn.Module):
    try:
        return model.named_modules(remove_duplicate=False)
    except TypeError:
        return model.named_modules()


def _embedding_name_allowed(name: str) -> bool:
    low = name.lower()
    return not any(x in low for x in ("position", "pos_embed", "wpe", "time"))


def _looks_like_token_embedding(name: str) -> bool:
    low = name.lower()
    return _embedding_name_allowed(name) and ("embed" in low or "wte" in low)


def _parameter_records(model: nn.Module):
    """Return one record per unique trainable Parameter, preserving aliases."""
    owners: Dict[int, List[Tuple[str, nn.Module, str]]] = defaultdict(list)
    params: Dict[int, nn.Parameter] = {}
    order: List[int] = []

    for module_name, module in _named_modules_all(model):
        for local_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid not in params:
                params[pid] = param
                order.append(pid)
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            owners[pid].append((full_name, module, local_name))

    # Defensive fallback for unusual modules that expose trainable parameters
    # through named_parameters but not recurse=False ownership traversal.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid not in params:
            params[pid] = param
            order.append(pid)
            owners[pid].append((name, model, name.rsplit(".", 1)[-1]))

    records = []
    for pid in order:
        entries = owners[pid]
        aliases = tuple(x[0] for x in entries)
        owner_modules = tuple(x[1] for x in entries)
        local_names = tuple(x[2] for x in entries)

        embedding_aliases = tuple(
            alias
            for alias, module, _ in entries
            if (
                (isinstance(module, nn.Embedding) and _embedding_name_allowed(alias))
                or _looks_like_token_embedding(alias)
            )
        )
        lm_head_aliases = tuple(alias for alias in aliases if "lm_head" in alias.lower())

        records.append({
            "param": params[pid],
            "aliases": aliases,
            "owners": owner_modules,
            "local_names": local_names,
            "embedding_aliases": embedding_aliases,
            "lm_head_aliases": lm_head_aliases,
        })
    return records


def _base_role(record: Mapping[str, Any]) -> str:
    p = record["param"]
    owners = record["owners"]
    local_names = record["local_names"]

    if p.ndim <= 1 or any(isinstance(m, _NORM_TYPES) for m in owners) or any(
        name.lower() == "bias" for name in local_names
    ):
        return "one_d"

    # A tied token-embedding / LM-head Parameter is intentionally routed as
    # embedding because the embedding state is the more conservative regime.
    if record["embedding_aliases"]:
        return "embedding"

    if record["lm_head_aliases"]:
        return "lm_head"

    if p.ndim == 2:
        return "matrix"
    return "nd"


def _matches_targets(aliases: Sequence[str], targets: Sequence[str]) -> bool:
    names = tuple(alias.lower() for alias in aliases)
    return any(any(target.lower() in name for name in names) for target in targets)


def _target_match_for_record(
    record: Mapping[str, Any],
    role: str,
    targets: Optional[Sequence[str]],
    explicit_targets: bool,
) -> Tuple[bool, bool]:
    """Return (matches, ignored_tied_lm_head_only_match)."""
    if targets is None:
        return True, False
    if not targets:
        return False, False

    aliases = record["aliases"]
    match = _matches_targets(aliases, targets)
    if not match:
        return False, False

    # If embedding and LM head share one Parameter, the shared tensor keeps the
    # embedding identity. An explicit lm_head-only target therefore must not
    # silently pull the shared embedding into APOLLO.
    if explicit_targets and role == "embedding" and record["lm_head_aliases"]:
        embedding_match = _matches_targets(record["embedding_aliases"], targets)
        if not embedding_match:
            return False, True

    return True, False


def _apollo_shape_ok(param: nn.Parameter, rank: int, min_dim_multiplier: int) -> bool:
    return rank > 0 and param.ndim == 2 and min(param.shape) >= rank * min_dim_multiplier


def _validate_route_options(route: str, opts: Mapping[str, Any]):
    if "stochastic_round_writeback" in opts and not isinstance(opts["stochastic_round_writeback"], bool):
        raise TypeError(f"{route}_stochastic_round_writeback must be bool")
    if "m_quant_type" in opts:
        value = opts["m_quant_type"]
        if value not in _M_TYPES:
            raise ValueError(
                f"Invalid {route}_m_quant_type={value!r}; expected one of {sorted(_M_TYPES)}"
            )

    if "v_quant_type" in opts:
        value = opts["v_quant_type"]
        if value not in _VC_TYPES:
            raise ValueError(
                f"Invalid {route}_v_quant_type={value!r}; expected one of {sorted(_VC_TYPES)}"
            )

    if "c_quant_type" in opts:
        value = opts["c_quant_type"]
        if value is not None and value not in _VC_TYPES:
            raise ValueError(
                f"Invalid {route}_c_quant_type={value!r}; expected None or one of {sorted(_VC_TYPES)}"
            )

    if "lr" in opts:
        value = opts["lr"]
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid {route}_lr={value!r}; expected a number >= 0")

    if "quantize" in opts:
        value = opts["quantize"]
        if not isinstance(value, bool) and not (
            isinstance(value, str) and value.lower() == "auto"
        ):
            raise TypeError(f"{route}_quantize must be True, False, or 'auto'")

    if "block_size" in opts:
        value = opts["block_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 128 or value % 128 != 0:
            raise ValueError(f"{route}_block_size must be >= 128 and a multiple of 128")

    if "m_block_size" in opts:
        value = opts["m_block_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 32 or value % 4 != 0:
            raise ValueError(f"{route}_m_block_size must be >= 32 and a multiple of 4")

    if "min_8bit_size" in opts:
        value = opts["min_8bit_size"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{route}_min_8bit_size must be an integer >= 0")


def _split_overrides(overrides: Dict[str, Any]):
    algorithm: Dict[str, Any] = {}
    route: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for key, value in overrides.items():
        if key in _COMMON_OVERRIDE_KEYS or key in _PRIMARY_OVERRIDE_KEYS:
            algorithm[key] = value
            continue

        for suffix in _ROUTE_OPTION_SUFFIXES:
            marker = "_" + suffix
            if key.endswith(marker):
                route_name = key[:-len(marker)]
                if route_name not in _ROUTE_NAMES:
                    raise TypeError(
                        f"Unknown route prefix {route_name!r} in {key!r}. "
                        f"Known routes: {sorted(_ROUTE_NAMES)}"
                    )
                route[route_name][suffix] = value
                break
        else:
            if key == "factored":
                raise TypeError(
                    "build_param_groups() owns 'factored' per route; edit the returned "
                    "group dict for expert overrides."
                )
            raise TypeError(f"Unknown build_param_groups() override: {key!r}")

    for route_name, opts in route.items():
        _validate_route_options(route_name, opts)

    return algorithm, dict(route)


def _semantic_dicts(spec: PresetSpec, algorithm_overrides: Mapping[str, Any]):
    primary = dict(spec.primary)
    fallback = dict(spec.fallback if spec.fallback is not None else spec.primary)

    for key, value in algorithm_overrides.items():
        if key in _COMMON_OVERRIDE_KEYS:
            primary[key] = value
            fallback[key] = value
        else:
            primary[key] = value

    return primary, fallback


def _fullrank_style(semantics: Mapping[str, Any], *, drop_momentum: bool = False):
    """Apply G2 full-rank fallback semantics without changing beta2."""
    out = dict(semantics)
    out.update({
        "factored": False,
        "scale_parameter": False,
        "d": 0.0,
        "beta3": None,
    })
    if drop_momentum:
        out["beta1"] = None
    return out


def _auto_quantize(
    preset: str, grouping: str, route: str, *, auto_profile: bool
) -> Optional[bool]:
    """Static evidence-backed policy; None means inherit optimizer global.

    Explicit G1 preserves the paper/reproduction protection profile.  The
    default grouping="auto" uses the same G1 topology with a more practical
    preset-aware quantization policy.
    """
    if grouping == "G1":
        if route == "one_d":
            return False
        if not auto_profile and route in {"embedding", "lm_head"}:
            return False
        if auto_profile and route in {"embedding", "lm_head"}:
            # Factored Adafactor shows the strongest token-embedding regime in
            # the trace, and its output-head statistics are also substantially
            # wider than attention/MLP.  RMSProp has no matching route-specific
            # evidence, so both routes stay conservative there too.
            if preset in {"adafactor", "rmsprop"}:
                return False
            # AdamW, CAME, and the Adam-style APOLLO fallback inherit the global
            # quantization choice.  This is a static recommendation, not a
            # runtime trace or a claim of universal optimality.
            return None

    if grouping == "G2" and route in {"one_d", "embedding"}:
        return False
    return None


def _resolved_quantize_option(
    preset: str,
    grouping: str,
    route: str,
    route_options: Mapping[str, Mapping[str, Any]],
    *,
    auto_profile: bool,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    opts = dict(route_options.get(route, {}))
    explicit = opts.pop("quantize", "auto")
    if isinstance(explicit, str):
        # Validation guarantees only "auto" reaches here.
        return _auto_quantize(preset, grouping, route, auto_profile=auto_profile), opts
    return explicit, opts


def _group(
    name: str,
    params: Iterable[nn.Parameter],
    semantics: Mapping[str, Any],
    route_options: Mapping[str, Mapping[str, Any]],
    *,
    preset: str,
    grouping: str,
    auto_profile: bool,
    weight_decay_zero: bool = False,
    apollo_rank: Optional[int] = None,
):
    params = list(params)
    if not params:
        return None

    group: Dict[str, Any] = {
        "params": params,
        "group_name": name,
        **dict(semantics),
        "apollo_rank": 0 if apollo_rank is None else int(apollo_rank),
    }

    if weight_decay_zero:
        group["weight_decay"] = 0.0

    raw_options = route_options.get(name, {})
    quantize_value, remaining_options = _resolved_quantize_option(
        preset, grouping, name, route_options, auto_profile=auto_profile
    )
    if quantize_value is not None:
        group["quantize"] = quantize_value

    precision_keys = {
        "m_quant_type", "v_quant_type", "c_quant_type",
        "min_8bit_size", "m_block_size", "block_size",
    }
    explicit_quantize = raw_options.get("quantize", "auto")
    if (
        quantize_value is False
        and isinstance(explicit_quantize, str)
        and any(key in raw_options for key in precision_keys)
    ):
        warnings.warn(
            f"Route {name!r} is unquantized by its auto policy, so explicit "
            "precision/block options are inactive unless "
            f"{name}_quantize=True is also set.",
            UserWarning,
            stacklevel=3,
        )

    # Explicit route options are the final builder-level override.
    group.update(remaining_options)

    if group.get("quantize", True):
        if group.get("beta1") is None and any(
            key in raw_options for key in ("m_quant_type", "m_block_size")
        ):
            warnings.warn(
                f"Route {name!r} has beta1=None, so its explicit M-state "
                "quantization options are currently inactive.",
                UserWarning,
                stacklevel=3,
            )
        if group.get("beta3") is None and "c_quant_type" in raw_options:
            warnings.warn(
                f"Route {name!r} has beta3=None, so its explicit C-state "
                "quantization option is currently inactive.",
                UserWarning,
                stacklevel=3,
            )

    return group


def _resolve_grouping(spec: PresetSpec, grouping: str) -> Tuple[str, bool]:
    if grouping is None:
        raise TypeError("grouping=None is ambiguous; use 'auto' or G0/G1/G2")
    if isinstance(grouping, str) and grouping.lower() == "auto":
        return spec.default_grouping, True
    key = _GROUPING_ALIASES.get(str(grouping))
    if key is None:
        raise ValueError(
            "Unknown grouping {!r}; use 'auto', G0/bare, G1/protected or G2/topology.".format(grouping)
        )
    return key, False


def _resolve_targets(spec: PresetSpec, value: Union[str, Sequence[str]]):
    if isinstance(value, str):
        if value.lower() != "auto":
            raise TypeError("apollo_targets must be 'auto' or a sequence such as ['attn', 'mlp']")
        if spec.default_apollo_targets is None:
            return None, False  # shape/topology-only auto targeting
        return tuple(spec.default_apollo_targets), False

    if value is None:
        raise TypeError("apollo_targets=None is ambiguous; use 'auto' or an explicit sequence")

    targets = tuple(value)
    if any(not isinstance(x, str) or not x for x in targets):
        raise TypeError("apollo_targets entries must be non-empty strings")
    return targets, True


def _resolve_multiplier(spec: PresetSpec, value: Union[str, int]) -> int:
    if isinstance(value, str) and value.lower() == "auto":
        return spec.default_apollo_min_dim_multiplier
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("apollo_min_dim_multiplier must be 'auto' or an integer >= 1")
    return value


def _warn_tied_route_overrides(
    records: Sequence[Mapping[str, Any]],
    route_options: Mapping[str, Mapping[str, Any]],
):
    if "lm_head" not in route_options:
        return False
    if not any(r["embedding_aliases"] and r["lm_head_aliases"] for r in records):
        return False

    warnings.warn(
        "A token embedding and lm_head share the same Parameter. The shared tensor "
        "is routed as 'embedding', so lm_head_* overrides do not apply to that tied "
        "Parameter; any untied lm_head parameters still use the lm_head route.",
        UserWarning,
        stacklevel=3,
    )
    return True


def build_param_groups(
    model: nn.Module,
    preset: str,
    grouping: str = "auto",
    *,
    apollo_rank: Optional[int] = None,
    apollo_targets: Union[str, Sequence[str]] = "auto",
    apollo_min_dim_multiplier: Union[str, int] = "auto",
    nd_factored: Optional[bool] = None,
    policy: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> List[Dict[str, Any]]:
    """Build ordinary mutable PyTorch parameter-group dictionaries.

    ``grouping='auto'`` and ``apollo_targets='auto'`` use static preset policies.
    Route-specific ``*_quantize='auto'`` similarly uses the selected grouping's
    evidence-backed default; it does not inspect optimizer states at runtime.
    Optional ``policy`` supplies defaults, custom presets, and partial route
    mappings. Precedence is generated groups, policy defaults, mapped preset,
    then mapped options. The function never changes parameter values or dtypes.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("build_param_groups() expects an nn.Module, not model.parameters()")

    preset_key = str(preset).lower()
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {', '.join(_preset_names())}")
    spec = PRESETS[preset_key]
    grouping_key, auto_profile = _resolve_grouping(spec, grouping)

    if nd_factored is not None:
        if not isinstance(nd_factored, bool):
            raise TypeError("nd_factored must be bool or None")
        if grouping_key != "G2":
            raise ValueError("nd_factored is only meaningful with G2/topology grouping")
        if nd_factored and preset_key not in {"adafactor", "came"}:
            raise ValueError(
                f"nd_factored=True is not part of the {preset_key!r} preset semantics; "
                "use adafactor/came or edit the returned nd group explicitly."
            )

    algorithm_overrides, route_options = _split_overrides(dict(overrides))
    primary, fallback = _semantic_dicts(spec, algorithm_overrides)

    is_apollo_preset = spec.default_apollo_rank > 0
    if not is_apollo_preset:
        if apollo_rank is not None:
            raise ValueError(f"apollo_rank requires an APOLLO-capable preset; got {preset_key!r}")
        if apollo_targets != "auto":
            raise ValueError(f"apollo_targets requires an APOLLO-capable preset; got {preset_key!r}")
        if apollo_min_dim_multiplier != "auto":
            raise ValueError(
                f"apollo_min_dim_multiplier requires an APOLLO-capable preset; got {preset_key!r}"
            )
        rank = 0
        targets: Optional[Tuple[str, ...]] = ()
        explicit_targets = False
        min_dim_multiplier = 1
    else:
        if apollo_rank is None:
            rank = spec.default_apollo_rank
        else:
            if not isinstance(apollo_rank, int) or isinstance(apollo_rank, bool):
                raise TypeError("apollo_rank must be an integer >= 0 or None")
            rank = apollo_rank
        if rank < 0:
            raise ValueError("apollo_rank must be >= 0")
        targets, explicit_targets = _resolve_targets(spec, apollo_targets)
        min_dim_multiplier = _resolve_multiplier(spec, apollo_min_dim_multiplier)
        if rank == 0:
            if explicit_targets and targets:
                warnings.warn(
                    "apollo_rank=0 disables APOLLO; explicit apollo_targets are ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            targets = ()

    if policy is not None:
        from .config import apply_policy

    records = _parameter_records(model)
    if not records:
        return apply_policy([], policy) if policy is not None else []

    tied_lm_override_warned = _warn_tied_route_overrides(records, route_options)
    buckets: Dict[str, List[nn.Parameter]] = defaultdict(list)
    warned_tied_lm_target = False

    for record in records:
        param = record["param"]
        role = _base_role(record)

        target_match, tied_lm_only = _target_match_for_record(
            record, role, targets, explicit_targets
        )
        if tied_lm_only and not warned_tied_lm_target:
            warnings.warn(
                "apollo_targets matched only the lm_head alias of a tied embedding/lm_head "
                "Parameter. The shared tensor is treated as 'embedding' and remains on the "
                "embedding route unless an embedding alias is explicitly targeted.",
                UserWarning,
                stacklevel=2,
            )
            warned_tied_lm_target = True

        shape_ok = _apollo_shape_ok(param, rank, min_dim_multiplier)
        protected_role = role in {"one_d", "embedding", "lm_head"}
        explicit_protected_target = explicit_targets and role in {"embedding", "lm_head"}
        use_apollo = (
            is_apollo_preset
            and target_match
            and shape_ok
            and (not protected_role or explicit_protected_target)
        )

        if grouping_key == "G0":
            if is_apollo_preset:
                buckets["apollo" if use_apollo else "fallback"].append(param)
            else:
                buckets["all"].append(param)

        elif grouping_key == "G1":
            if use_apollo:
                buckets["apollo"].append(param)
            elif protected_role:
                # G1 remains a protection policy, but the three protected
                # classes stay separate so production users can override them
                # independently without switching to G2 algorithm semantics.
                buckets[role].append(param)
            else:
                buckets["other"].append(param)

        else:  # G2
            buckets["apollo" if use_apollo else role].append(param)

    groups: List[Dict[str, Any]] = []

    if grouping_key == "G0":
        if is_apollo_preset:
            group = _group(
                "apollo", buckets["apollo"], primary, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, apollo_rank=rank,
            )
            if group:
                groups.append(group)
            group = _group(
                "fallback", buckets["fallback"], fallback, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile,
            )
            if group:
                groups.append(group)
        else:
            group = _group(
                "all", buckets["all"], primary, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile,
            )
            if group:
                groups.append(group)

    elif grouping_key == "G1":
        non_apollo = fallback if is_apollo_preset else primary

        for role in ("one_d", "embedding", "lm_head"):
            group = _group(
                role, buckets[role], non_apollo, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, weight_decay_zero=True,
            )
            if group:
                groups.append(group)

        if is_apollo_preset:
            group = _group(
                "apollo", buckets["apollo"], primary, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, apollo_rank=rank,
            )
            if group:
                groups.append(group)

        group = _group(
            "other", buckets["other"], non_apollo, route_options,
            preset=preset_key, grouping=grouping_key, auto_profile=auto_profile,
        )
        if group:
            groups.append(group)

    else:  # G2
        base = fallback if is_apollo_preset else primary

        group = _group(
            "one_d", buckets["one_d"], base, route_options,
            preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, weight_decay_zero=True,
        )
        if group:
            groups.append(group)

        group = _group(
            "embedding", buckets["embedding"], _fullrank_style(base, drop_momentum=True),
            route_options, preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, weight_decay_zero=True,
        )
        if group:
            groups.append(group)

        group = _group(
            "lm_head", buckets["lm_head"], _fullrank_style(base), route_options,
            preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, weight_decay_zero=True,
        )
        if group:
            groups.append(group)

        if is_apollo_preset:
            group = _group(
                "apollo", buckets["apollo"], primary, route_options,
                preset=preset_key, grouping=grouping_key, auto_profile=auto_profile, apollo_rank=rank,
            )
            if group:
                groups.append(group)

        group = _group(
            "matrix", buckets["matrix"], base, route_options,
            preset=preset_key, grouping=grouping_key, auto_profile=auto_profile,
        )
        if group:
            groups.append(group)

        use_nd_factored = spec.default_nd_factored if nd_factored is None else nd_factored
        nd_semantics = dict(base) if use_nd_factored else _fullrank_style(base)
        group = _group(
            "nd", buckets["nd"], nd_semantics, route_options,
            preset=preset_key, grouping=grouping_key, auto_profile=auto_profile,
        )
        if group:
            groups.append(group)

    assigned = [param for group in groups for param in group["params"]]
    assigned_ids = {id(param) for param in assigned}
    expected_ids = {id(record["param"]) for record in records}
    if len(assigned) != len(assigned_ids) or assigned_ids != expected_ids:
        raise RuntimeError("Internal routing error: parameters were duplicated or omitted")

    generated = {group["group_name"] for group in groups}
    for route_name in route_options:
        if route_name in generated:
            continue
        if route_name == "lm_head" and tied_lm_override_warned:
            # The dedicated tied-weight warning is more informative.
            continue
        warnings.warn(
            f"Route override for {route_name!r} did not match a generated group.",
            UserWarning,
            stacklevel=2,
        )

    if policy is not None:
        groups = apply_policy(groups, policy)
    return groups
