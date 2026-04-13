from __future__ import annotations

from typing import Iterable, List


def round_root(root: float, integer_tol: float = 1e-6) -> float | int:
    nearest = round(root)
    if abs(root - nearest) < integer_tol:
        return int(nearest)
    return round(root, 6)


def deduplicate(roots: Iterable[float], tol: float = 1e-4) -> List[float]:
    unique: List[float] = []
    for root in roots:
        if not any(abs(root - current) < tol for current in unique):
            unique.append(root)
    return unique
