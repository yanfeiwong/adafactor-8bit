# Copyright (c) 2026 WANG YAN
# Licensed under the MIT License.

"""Semantic preset definitions used by :mod:`adafactor8bit.routing`.

The public guided API is :func:`adafactor8bit.build_param_groups`.  Presets
contain only Adafactor8Bit path semantics; benchmark grids and third-party
optimizer definitions stay outside the package.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PresetSpec:
    name: str
    default_grouping: str
    primary: Dict[str, Any]
    fallback: Optional[Dict[str, Any]] = None
    default_apollo_rank: int = 0
    default_apollo_targets: Optional[Tuple[str, ...]] = None
    default_apollo_min_dim_multiplier: int = 1
    default_nd_factored: bool = False


# Presets define path semantics only.  Learning rate, weight decay, and
# quantization formats remain constructor-level deployment choices unless a
# route explicitly overrides them.
_SGD = {
    "beta1": None,
    "beta2": None,
    "beta2_decay": None,
    "beta3": None,
    "factored": False,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 0.0,
}

_ADAFACTOR = {
    "beta1": None,
    "beta2": None,
    "beta2_decay": -0.8,
    "beta3": None,
    "factored": True,
    "relative_step": False,
}

_ADAMW = {
    "beta1": 0.9,
    "beta2": 0.999,
    "beta2_decay": None,
    "beta3": None,
    "factored": False,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 0.0,
    "eps": (1e-8, 1e-3),
}

_RMSPROP = {
    "beta1": None,
    "beta2": 0.99,
    "beta2_decay": None,
    "beta3": None,
    "factored": False,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 0.0,
    "eps": (1e-8, 1e-3),
}

_CAME = {
    "beta1": 0.9,
    "beta2": 0.999,
    "beta2_decay": None,
    "beta3": 0.9999,
    "factored": True,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 1.0,
    "eps_came": 1e-16,
}

_APOLLO_FALLBACK = {
    # Non-target parameters use full-rank Adam-style semantics while retaining
    # the APOLLO path's epsilon convention.
    "beta1": 0.9,
    "beta2": 0.999,
    "beta2_decay": None,
    "beta3": None,
    "factored": False,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 0.0,
    # Keep the APOLLO epsilon convention on the fallback route.
    "eps": (1e-6, 1e-3),
}

_APOLLO = {
    "beta1": 0.9,
    "beta2": 0.999,
    "beta2_decay": None,
    "beta3": None,
    "factored": False,
    "relative_step": False,
    "scale_parameter": False,
    "scale_weight_decay": False,
    "d": 0.0,
    "eps": (1e-6, 1e-3),
    "apollo_eps": 1e-6,
    "apollo_factorize": False,
    "apollo_update_proj_gap": 200,
    "apollo_scale_type": "channel",
    "apollo_scale": 1.0,
    "apollo_scale_front": False,
    "apollo_cache_proj": True,
    "enable_fira_for_apollo": True,
}


PRESETS = {
    "sgd": PresetSpec(
        name="sgd",
        default_grouping="G1",
        primary=_SGD,
    ),
    "adafactor": PresetSpec(
        name="adafactor",
        default_grouping="G1",
        primary=_ADAFACTOR,
    ),
    "adamw": PresetSpec(
        name="adamw",
        default_grouping="G1",
        primary=_ADAMW,
    ),
    "rmsprop": PresetSpec(
        name="rmsprop",
        default_grouping="G1",
        primary=_RMSPROP,
    ),
    "came": PresetSpec(
        name="came",
        default_grouping="G1",
        primary=_CAME,
    ),
    "apollo": PresetSpec(
        name="apollo",
        default_grouping="G1",
        primary=_APOLLO,
        fallback=_APOLLO_FALLBACK,
        default_apollo_rank=256,
        default_apollo_targets=("attn", "mlp"),
        # Standard APOLLO uses rank eligibility; stricter topology selectors
        # can override this multiplier.
        default_apollo_min_dim_multiplier=1,
    ),
}


def _preset_names():
    return tuple(PRESETS.keys())
