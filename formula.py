"""Неизменяемые деревья формул и операции над ними. Python 3.10+."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Var:
    name: str


@dataclass(frozen=True, slots=True)
class Not:
    operand: "Formula"


@dataclass(frozen=True, slots=True)
class Implies:
    left: "Formula"
    right: "Formula"


Formula: TypeAlias = Var | Not | Implies


def format_formula(formula: Formula) -> str:
    """Однозначная запись: каждая импликация заключена в скобки."""
    if isinstance(formula, Var):
        return formula.name
    if isinstance(formula, Not):
        return "!" + format_formula(formula.operand)
    return f"({format_formula(formula.left)} -> {format_formula(formula.right)})"


def variables(formula: Formula) -> set[str]:
    """Имена переменных в дереве; обход не зависит от глубины рекурсии."""
    result: set[str] = set()
    pending = [formula]
    while pending:
        node = pending.pop()
        if isinstance(node, Var):
            result.add(node.name)
        elif isinstance(node, Not):
            pending.append(node.operand)
        else:
            pending.extend((node.left, node.right))
    return result


def substitute(formula: Formula, variable: str, replacement: Formula) -> Formula:
    """Одновременно заменить ВСЕ исходные вхождения ОДНОЙ переменной.

    Внутри replacement повторная замена не выполняется. Исходное дерево
    не меняется; общие неизменяемые поддеревья можно безопасно переиспользовать.
    """
    if isinstance(formula, Var):
        return replacement if formula.name == variable else formula
    if isinstance(formula, Not):
        return Not(substitute(formula.operand, variable, replacement))
    return Implies(
        substitute(formula.left, variable, replacement),
        substitute(formula.right, variable, replacement),
    )


def infer_substitution(
    source: Formula, target: Formula, variable: str
) -> Formula | None:
    """Найти X, для которого target = source[variable := X].

    При первом вхождении variable запоминаем соответствующее поддерево
    target. Все остальные вхождения обязаны дать то же поддерево.
    Остальные переменные и операции должны совпадать буквально.
    None означает неудачу или отсутствие variable в source.
    """
    replacement: Formula | None = None
    pending = [(source, target)]
    while pending:
        before, after = pending.pop()
        if isinstance(before, Var) and before.name == variable:
            if replacement is None:
                replacement = after
            elif replacement != after:
                return None
        elif isinstance(before, Var):
            if before != after:
                return None
        elif isinstance(before, Not):
            if not isinstance(after, Not):
                return None
            pending.append((before.operand, after.operand))
        else:
            if not isinstance(after, Implies):
                return None
            # Стек LIFO: левая часть будет проверена первой.
            pending.append((before.right, after.right))
            pending.append((before.left, after.left))
    return replacement


def tree_depth(formula: Formula) -> int:
    """Глубина дерева: одиночная переменная имеет глубину 1."""
    result = 0
    pending = [(formula, 1)]
    while pending:
        node, depth = pending.pop()
        result = max(result, depth)
        if isinstance(node, Not):
            pending.append((node.operand, depth + 1))
        elif isinstance(node, Implies):
            pending.extend(((node.left, depth + 1), (node.right, depth + 1)))
    return result
