# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

"""Optional dictionary overrides for the existing parameter-group builder."""

from copy import deepcopy
from inspect import signature
from typing import Any, Dict, List, Mapping

from .optimizer import Adafactor8Bit
from .presets import PRESETS
from .routing import _ROUTE_NAMES, _validate_route_options

_ARGUMENTS = signature(Adafactor8Bit.__init__).parameters
_CONFIG_KEYS = set(_ARGUMENTS) - {"self", "params"}
_SEMANTIC_KEYS = {"apollo_rank"}
for _spec in PRESETS.values():
    _SEMANTIC_KEYS.update(_spec.primary)
    _SEMANTIC_KEYS.update(_spec.fallback or {})


def _validate_config(config: Mapping[str, Any], label: str) -> None:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise TypeError(f"Unknown optimizer options in {label}: {sorted(unknown)}")
    _validate_route_options(label, config)


def _resolve_preset(name: str, custom: Mapping[str, Any], trail=()) -> Dict[str, Any]:
    if not isinstance(name, str):
        raise TypeError("Policy preset names must be strings")
    if name in PRESETS:
        spec = PRESETS[name]
        result = {k: deepcopy(_ARGUMENTS[k].default) for k in _SEMANTIC_KEYS}
        result.update(deepcopy(spec.primary))
        result["apollo_rank"] = spec.default_apollo_rank
        return result
    if name not in custom:
        raise ValueError(f"Unknown policy preset {name!r}")
    if name in trail:
        raise ValueError(f"Cyclic policy preset inheritance: {' -> '.join((*trail, name))}")
    definition = dict(custom[name])
    parent = definition.pop("preset", None)
    if parent is None:
        raise ValueError(f"Custom preset {name!r} requires a base 'preset'")
    result = _resolve_preset(parent, custom, (*trail, name))
    result.update(deepcopy(definition))
    return result


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate a partial policy without inspecting or modifying a model."""
    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping")
    unknown = set(policy) - {"defaults", "presets", "mapping"}
    if unknown:
        raise TypeError(f"Unknown policy sections: {sorted(unknown)}")
    _validate_config(policy.get("defaults", {}), "policy.defaults")
    custom = policy.get("presets", {})
    mapping = policy.get("mapping", {})
    if not isinstance(custom, Mapping) or not isinstance(mapping, Mapping):
        raise TypeError("policy.presets and policy.mapping must be mappings")
    for name, definition in custom.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Custom preset names must be non-empty strings")
        if name in PRESETS:
            raise ValueError(f"Custom preset {name!r} shadows a built-in; choose a distinct name")
        if not isinstance(definition, Mapping):
            raise TypeError(f"Custom preset {name!r} must be a mapping")
        _validate_config({k: v for k, v in definition.items() if k != "preset"}, f"presets.{name}")
    for name in custom:
        _resolve_preset(name, custom)
    for route, entry in mapping.items():
        if route not in _ROUTE_NAMES:
            raise ValueError(f"Unknown policy route {route!r}; expected one of {sorted(_ROUTE_NAMES)}")
        if not isinstance(entry, Mapping):
            raise TypeError(f"mapping.{route} must be a mapping")
        _validate_config({k: v for k, v in entry.items() if k != "preset"}, f"mapping.{route}")
        if "preset" in entry:
            _resolve_preset(entry["preset"], custom)


def apply_policy(groups: List[Dict[str, Any]], policy: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Overlay configuration while retaining parameter identities and order.

    Precedence: generated group < policy defaults < mapped preset < mapped
    options. Known routes absent from a model are allowed, so pruning a model
    does not require deleting unrelated mapping entries.
    """
    validate_policy(policy)
    result = []
    for group in groups:
        resolved = dict(group)
        resolved.update(deepcopy(dict(policy.get("defaults", {}))))
        entry = dict(policy.get("mapping", {}).get(group["group_name"], {}))
        preset = entry.pop("preset", None)
        if preset is not None:
            resolved.update(_resolve_preset(preset, policy.get("presets", {})))
        resolved.update(deepcopy(entry))
        rank = resolved.get("apollo_rank", 0)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("apollo_rank must be an integer >= 0")
        if rank and any(p.ndim != 2 or min(p.shape) < rank for p in resolved["params"]):
            raise ValueError(f"Route {group['group_name']!r} contains parameters ineligible for APOLLO rank {rank}")
        result.append(resolved)
    return result
