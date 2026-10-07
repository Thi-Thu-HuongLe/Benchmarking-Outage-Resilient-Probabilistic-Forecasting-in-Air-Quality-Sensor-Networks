"""Compatibility entry point for environments installed before the artifact cleanup."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "The legacy multi-dataset CLI was retired. Run the explicit EPA AQS scripts "
        "documented in Code/README.md."
    )
