"""Base classes for the TOD step configurations.

Every TOD operation owns an immutable config object that unpacks its parameter block, applies its
defaults, validates them, and answers whether the step runs on a given iteration. `StepConfig` is
what they share; `CGConfig` is the conjugate-gradient block that both the correlated-noise step and
the CG mapmaker embed. The step configs themselves live with their steps.
"""
from dataclasses import dataclass, fields
from typing import Self

import numpy as np
from pixell.bunch import Bunch


# Each config class owns the parameter names, defaults, and validation for one TOD operation.
# ``process_tod`` still states the physical execution order explicitly. Correlated noise and data
# selection stay inside the mapmaker scan loops because their position relative to sigma0,
# diagnostics, vetoes, and map accumulation is part of the algorithm.
@dataclass(frozen=True)
class StepConfig:
    """Common parameter construction and iteration gate for a TOD step."""

    enabled: bool = False
    from_iter: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false.")
        if not isinstance(self.from_iter, int) or isinstance(self.from_iter, bool):
            raise ValueError("from_iter must be an integer.")
        if self.from_iter < 1:
            raise ValueError("from_iter must be at least 1.")

    def is_active(self, iteration: int) -> bool:
        """Whether this step runs in the given Gibbs iteration."""
        return self.enabled and iteration >= self.from_iter

    @classmethod
    def _from_block(cls, block_name: str, block: Bunch | dict, **resolved_values) -> Self:
        """Construct this step config and reject fields it does not own."""
        try:
            values = dict(block)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{block_name}' must be a parameter block.") from error
        constructor_fields = {item.name for item in fields(cls) if item.init}
        parameter_fields = constructor_fields - set(resolved_values)
        unknown = sorted(set(values) - parameter_fields)
        if unknown:
            raise ValueError(f"Unknown key(s) {unknown} in '{block_name}'. That block accepts "
                             f"{sorted(parameter_fields)}.")
        try:
            return cls(**values, **resolved_values)
        except TypeError as error:
            raise ValueError(f"Invalid '{block_name}' configuration: {error}") from error


@dataclass(frozen=True)
class CGConfig:
    """Conjugate-gradient controls shared by the two CG uses in TOD processing."""

    max_iter: int = 0
    err_tol: float = 1.0e-4

    def __post_init__(self) -> None:
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool):
            raise ValueError("max_iter must be an integer.")
        if self.max_iter < 0:
            raise ValueError("max_iter cannot be negative.")
        if not np.isfinite(self.err_tol) or self.err_tol < 0:
            raise ValueError("err_tol must be a finite, non-negative number.")

    @classmethod
    def from_block(cls, block_name: str, block: Bunch | dict,
                   require_all: bool = False) -> "CGConfig":
        """Build CG controls and optionally require both fields to be stated."""
        try:
            values = dict(block)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{block_name}' must be a parameter block.") from error
        if require_all:
            missing = sorted({"max_iter", "err_tol"} - set(values))
            if missing:
                raise ValueError(f"Missing required key(s) {missing} in '{block_name}'.")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid '{block_name}' configuration: {error}") from error
