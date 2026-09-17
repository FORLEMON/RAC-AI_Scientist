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
            "runtime_communication": self.R1,
            "runtime_routing": self.R2,
            "work_contracts": self.R3,
            "verifier": self.R4,
            "recovery": self.R5,
        }
        try:
            return self >= thresholds[component]
        except KeyError as exc:
            raise ValueError(f"unknown RAC component {component!r}") from exc
