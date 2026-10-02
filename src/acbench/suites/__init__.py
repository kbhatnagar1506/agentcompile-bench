"""Suite registry. Each adapter converts one benchmark to the shared task format."""

from __future__ import annotations

from .base import Suite


def get_suite(name: str) -> Suite:
    if name == "toy":
        from .toy import ToySuite

        return ToySuite()
    if name == "tau-retail":
        from .tau import TauRetail

        return TauRetail()
    raise ValueError(f"unknown suite {name!r}; known: {SUITES}")


SUITES = ("toy", "tau-retail")
