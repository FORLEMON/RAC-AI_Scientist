from __future__ import annotations

from enum import IntEnum


class Condition(IntEnum):
    """Cumulative experimental conditions."""

    N0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5

    @classmethod
    def parse(cls, value: str | "Condition") -> "Condition":
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            allowed = ", ".join(item.name for item in cls)
            raise ValueError(f"unknown condition {value!r}; expected one of {allowed}") from exc

    def enables(self, component: str) -> bool:
        thresholds = {
            "runtime_routing": self.R1,
            "work_contracts": self.R2,
            "verifier": self.R3,
            "recovery": self.R4,
            "issue_aware_control": self.R5,
        }
        try:
            return self >= thresholds[component]
        except KeyError as exc:
            raise ValueError(f"unknown RAC component {component!r}") from exc
