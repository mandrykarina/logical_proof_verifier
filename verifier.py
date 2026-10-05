"""Проверка одного шага в исчислении варианта 2: MP и атомарная β."""

from dataclasses import dataclass
from typing import Literal

from formula import Formula, Implies, infer_substitution, variables
from parser import parse_formula


@dataclass(frozen=True, slots=True)
class Entry:
    label: str
    formula: Formula
    rule: Literal["axiom", "mp", "beta"]
    sources: tuple[str, ...] = ()
    variable: str | None = None
    replacement: Formula | None = None


AXIOMS = (
    Entry("A1", parse_formula("p -> (q -> p)"), "axiom"),
    Entry("A2", parse_formula("(s -> (p -> q)) -> ((s -> p) -> (s -> q))"), "axiom"),
    Entry("A3", parse_formula("(!p -> !q) -> (q -> p)"), "axiom"),
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    status: Literal["accepted", "known", "rejected"]
    formula: Formula
    entry: Entry | None = None


class ProofSession:
    """Известны только конкретные аксиомы и успешно проверенные шаги.

    Произвольные экземпляры схем аксиом автоматически НЕ принимаются:
    одна строка доказательства допускает только одну β или одно MP.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._steps: list[Entry] = []
        self._known: dict[Formula, Entry] = {entry.formula: entry for entry in AXIOMS}
        self._by_label: dict[str, Entry] = {entry.label: entry for entry in AXIOMS}
        self.goal: Formula | None = None

    @property
    def steps(self) -> tuple[Entry, ...]:
        return tuple(self._steps)

    @property
    def goal_reached(self) -> bool:
        return self.goal is not None and self.goal in self._known

    def known_entry(self, formula: Formula) -> Entry | None:
        return self._known.get(formula)

    def get_entry(self, label: str) -> Entry:
        return self._by_label[label]

    def set_goal(self, formula: Formula) -> None:
        # Цель не является посылкой и не попадает в _known.
        self.goal = formula

    def submit(self, formula: Formula) -> CheckResult:
        existing = self._known.get(formula)
        if existing is not None:
            return CheckResult("known", formula, existing)

        # Один MP: точное структурное совпадение B и отдельно известное A.
        for source in self._known.values():
            implication = source.formula
            if isinstance(implication, Implies) and implication.right == formula:
                premise = self._known.get(implication.left)
                if premise is not None:
                    return self._accept(
                        formula, "mp", (source.label, premise.label)
                    )

        # Одна β из одной известной формулы. Никаких скрытых промежуточных шагов.
        for source in self._known.values():
            for variable in sorted(variables(source.formula)):
                replacement = infer_substitution(source.formula, formula, variable)
                if replacement is not None:
                    return self._accept(
                        formula, "beta", (source.label,), variable, replacement
                    )
        return CheckResult("rejected", formula)

    def _accept(
        self,
        formula: Formula,
        rule: Literal["mp", "beta"],
        sources: tuple[str, ...],
        variable: str | None = None,
        replacement: Formula | None = None,
    ) -> CheckResult:
        entry = Entry(str(len(self._steps) + 1), formula, rule, sources, variable, replacement)
        self._steps.append(entry)
        self._known[formula] = entry
        self._by_label[entry.label] = entry
        return CheckResult("accepted", formula, entry)
