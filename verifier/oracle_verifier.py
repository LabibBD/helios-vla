"""Simulator-state outcome verifier for HELIOS.

IMPORTANT:
This verifier uses LIBERO simulator ground truth.
It is the robust MVP verifier used for recovery experiments.
It must NOT be described as a visual verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OracleVerification:
    success: bool
    source: str = "libero_check_success"


class SimulatorOracleVerifier:
    """Extract LIBERO's ground-truth task-success condition."""

    @staticmethod
    def _to_bool(value: Any) -> bool:
        array = np.asarray(value)
        if array.size == 0:
            return False
        return bool(array.reshape(-1)[0])

    def verify(self, info: dict[str, Any]) -> OracleVerification:
        # Gymnasium may move terminal information into final_info.
        final_info = info.get("final_info")

        if isinstance(final_info, dict) and "is_success" in final_info:
            return OracleVerification(
                success=self._to_bool(final_info["is_success"])
            )

        if "is_success" in info:
            return OracleVerification(
                success=self._to_bool(info["is_success"])
            )

        return OracleVerification(success=False)
