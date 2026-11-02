from __future__ import annotations


def filter_valid(rows: list[dict]) -> list[dict]:
    """Drop rows where the teacher produced no valid, in-schema label."""
    return [r for r in rows if r.get("label")]