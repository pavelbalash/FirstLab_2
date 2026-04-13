from __future__ import annotations

from typing import List, Tuple

from solver.methods import (
    cardano_method,
    chord_tangent_method,
    horner_method,
    linear_method,
    newton_method,
    secant_method,
    vieta_method,
)
from solver.utils import deduplicate, round_root


def read_coefficients() -> List[float]:
    while True:
        raw = input("Введите коэффициенты a b c d n m k через пробел: ").strip()
        parts = raw.split()

        if len(parts) != 7:
            print("надо 7 коэффициентов!")
            continue

        try:
            return [float(part) for part in parts]
        except ValueError:
            print("коэффициенты должны быть числами")


def solve_polynomial(coeffs: List[float]) -> Tuple[List[float], bool]:
    if all(abs(c) < 1e-12 for c in coeffs):
        print("Любое число является решением")
        return [], True

    coeffs = coeffs[:]
    while coeffs and abs(coeffs[0]) < 1e-12:
        coeffs.pop(0)

    if len(coeffs) == 1:
        print("Нет решений")
        return [], False

    result: List[float] = []

    while len(coeffs) > 1:
        degree = len(coeffs) - 1

        if degree == 1:
            result.extend(linear_method(coeffs))
            break

        if degree == 2:
            result.extend(vieta_method(coeffs))
            break

        if degree == 3:
            result.extend(cardano_method(coeffs))
            break

        root = None
        for method in (newton_method, chord_tangent_method, secant_method):
            root = method(coeffs)
            if root is not None:
                break

        if root is None:
            print("Не удалось найти следующий корень численными методами")
            break

        result.append(root)
        coeffs, remainder = horner_method(coeffs, root)

        if abs(remainder) > 1e-3:
            print("Предупреждение: деление дало заметный остаток")

    normalized = [round_root(root) for root in result]
    deduped = deduplicate(normalized)
    return deduped, False


def run() -> None:
    while True:
        coeffs = read_coefficients()
        roots, is_trivial = solve_polynomial(coeffs)

        if not is_trivial:
            print(f"Корни: {roots}")

        answer = input("Продолжить? (нет/что-то другое): ").strip().lower()
        if answer == "нет":
            print("Завершение программы.")
            break


if __name__ == "__main__":
    run()
