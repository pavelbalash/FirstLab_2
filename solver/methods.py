from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple


EPS = 1e-10
SEARCH_LEFT = -1000.0
SEARCH_RIGHT = 1000.0
SEARCH_STEP = 1.0
MAX_ITERS = 200


def poly_value(coeffs: List[float], x: float) -> float:
    value = 0.0
    for coefficient in coeffs:
        value = value * x + coefficient
    return value


def derivative_coeffs(coeffs: List[float]) -> List[float]:
    degree = len(coeffs) - 1
    return [coeffs[i] * (degree - i) for i in range(degree)]


def poly_derivative_value(coeffs: List[float], x: float) -> float:
    return poly_value(derivative_coeffs(coeffs), x)


def horner_method(coeffs: List[float], root: float) -> Tuple[List[float], float]:
    reduced = [coeffs[0]]
    for i in range(1, len(coeffs) - 1):
        reduced.append(reduced[-1] * root + coeffs[i])
    remainder = reduced[-1] * root + coeffs[-1]
    return reduced, remainder


def linear_method(coeffs: List[float]) -> List[float]:
    a, b = coeffs
    if abs(a) < EPS:
        return []
    return [-b / a]


def vieta_method(coeffs: List[float]) -> List[float]:
    a, b, c = coeffs
    if abs(a) < EPS:
        return linear_method([b, c])

    d = b * b - 4 * a * c
    if d < -EPS:
        return []
    if abs(d) <= EPS:
        return [-b / (2 * a)]

    sqrt_d = math.sqrt(d)
    return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]


def cardano_method(coeffs: List[float]) -> List[float]:
    a, b, c, d = coeffs
    if abs(a) < EPS:
        return vieta_method([b, c, d])

    p = (3 * a * c - b * b) / (3 * a * a)
    q = (2 * b**3 - 9 * a * b * c + 27 * a * a * d) / (27 * a**3)
    delta = (q / 2) ** 2 + (p / 3) ** 3

    roots: List[float] = []
    shift = -b / (3 * a)

    if delta > EPS:
        u = _cuberoot(-q / 2 + math.sqrt(delta))
        v = _cuberoot(-q / 2 - math.sqrt(delta))
        roots.append(u + v + shift)
        return roots

    if abs(delta) <= EPS:
        u = _cuberoot(-q / 2)
        roots.append(2 * u + shift)
        roots.append(-u + shift)
        return roots

    r = math.sqrt(-(p**3) / 27)
    phi = math.acos(max(-1.0, min(1.0, -q / (2 * r))))
    t = 2 * math.sqrt(-p / 3)
    for k in range(3):
        angle = (phi + 2 * math.pi * k) / 3
        roots.append(t * math.cos(angle) + shift)
    return roots


def newton_method(coeffs: List[float]) -> Optional[float]:
    for start in range(-10, 11):
        x = float(start)
        for _ in range(MAX_ITERS):
            fx = poly_value(coeffs, x)
            dfx = poly_derivative_value(coeffs, x)
            if abs(dfx) < EPS:
                break
            nx = x - fx / dfx
            if abs(nx - x) < 1e-9 and abs(poly_value(coeffs, nx)) < 1e-6:
                return nx
            x = nx
    return None


def chord_tangent_method(coeffs: List[float]) -> Optional[float]:
    bracket = _find_sign_change(coeffs)
    if bracket is None:
        return None

    left, right = bracket
    f_left = poly_value(coeffs, left)
    f_right = poly_value(coeffs, right)

    for _ in range(MAX_ITERS):
        if abs(f_right - f_left) < EPS:
            return None
        x = (left * f_right - right * f_left) / (f_right - f_left)
        fx = poly_value(coeffs, x)

        dfx = poly_derivative_value(coeffs, x)
        if abs(dfx) > EPS:
            tangent = x - fx / dfx
        else:
            tangent = x

        candidate = (x + tangent) / 2
        f_candidate = poly_value(coeffs, candidate)

        if abs(f_candidate) < 1e-6:
            return candidate

        if f_left * f_candidate <= 0:
            right, f_right = candidate, f_candidate
        else:
            left, f_left = candidate, f_candidate

    return None


def secant_method(coeffs: List[float]) -> Optional[float]:
    bracket = _find_sign_change(coeffs)
    if bracket is None:
        return None

    x0, x1 = bracket
    f0 = poly_value(coeffs, x0)
    f1 = poly_value(coeffs, x1)

    for _ in range(MAX_ITERS):
        if abs(f1 - f0) < EPS:
            return None

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = poly_value(coeffs, x2)

        if abs(f2) < 1e-6:
            return x2

        x0, f0 = x1, f1
        x1, f1 = x2, f2

    return None


def _find_sign_change(coeffs: List[float]) -> Optional[Tuple[float, float]]:
    x = SEARCH_LEFT
    prev = poly_value(coeffs, x)

    while x < SEARCH_RIGHT:
        nx = x + SEARCH_STEP
        curr = poly_value(coeffs, nx)

        if abs(curr) < 1e-8:
            return nx - SEARCH_STEP, nx
        if prev * curr < 0:
            return x, nx

        x = nx
        prev = curr

    return None


def _cuberoot(x: float) -> float:
    if x >= 0:
        return x ** (1 / 3)
    return -((-x) ** (1 / 3))
