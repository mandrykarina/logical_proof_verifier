"""Проверки синтаксиса, правил, состояния сеанса и консольного интерфейса.

Запуск из корня проекта: python -m unittest discover -s tests -v
"""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from itertools import product
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from formula import (
    Implies, Not, Var, format_formula, infer_substitution, substitute,
    tree_depth, variables,
)
from main import ConsoleApp
from parser import (
    FormulaSyntaxError, MAX_DEPTH, MAX_PARSE_DEPTH, MAX_INPUT_LENGTH, MAX_TOKENS,
    parse_formula, tokenize,
)
from verifier import AXIOMS, ProofSession


ROOT = Path(__file__).resolve().parents[1]
PROOF = (
    "(s -> (r -> q)) -> ((s -> r) -> (s -> q))",
    "(p -> (r -> q)) -> ((p -> r) -> (p -> q))",
    "(p -> (r -> p)) -> ((p -> r) -> (p -> p))",
    "p -> (r -> p)",
    "(p -> r) -> (p -> p)",
    "(p -> (q -> p)) -> (p -> p)",
    "p -> p",
)


def evaluate(formula, values):
    """Независимая семантическая проверка только для тестов."""
    if isinstance(formula, Var):
        return values[formula.name]
    if isinstance(formula, Not):
        return not evaluate(formula.operand, values)
    return not evaluate(formula.left, values) or evaluate(formula.right, values)


class FormulaTests(unittest.TestCase):
    def test_structural_equality_and_hash(self):
        a = Implies(Var("p"), Not(Var("q")))
        b = Implies(Var("p"), Not(Var("q")))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual({a: 1}[b], 1)

    def test_different_structure_is_not_equal(self):
        self.assertNotEqual(parse_formula("p -> q"), parse_formula("q -> p"))
        self.assertNotEqual(parse_formula("p"), parse_formula("!!p"))
        self.assertNotEqual(parse_formula("!(p -> q)"), parse_formula("!p -> q"))

    def test_immutable_nodes(self):
        node = Var("p")
        with self.assertRaises(FrozenInstanceError):
            node.name = "q"

    def test_variables_and_depth(self):
        formula = parse_formula("p -> (!q -> p)")
        self.assertEqual(variables(formula), {"p", "q"})
        self.assertEqual(tree_depth(formula), 4)

    def test_substitute_all_occurrences(self):
        original = parse_formula("p -> (q -> p)")
        result = substitute(original, "p", parse_formula("!r"))
        self.assertEqual(result, parse_formula("!r -> (q -> !r)"))
        self.assertEqual(original, parse_formula("p -> (q -> p)"))

    def test_substitution_does_not_recurse_into_inserted_formula(self):
        original = parse_formula("p -> (q -> p)")
        result = substitute(original, "p", parse_formula("p -> r"))
        self.assertEqual(result, parse_formula("(p -> r) -> (q -> (p -> r))"))

    def test_missing_variable(self):
        source = parse_formula("p -> p")
        self.assertEqual(substitute(source, "z", Var("q")), source)
        self.assertIsNone(infer_substitution(source, source, "z"))

    def test_infer_var_negation_and_compound_replacements(self):
        source = parse_formula("p -> (q -> p)")
        for replacement in ("r", "!r", "r -> s", "p -> r", "p", "q"):
            with self.subTest(replacement=replacement):
                x = parse_formula(replacement)
                target = substitute(source, "p", x)
                self.assertEqual(infer_substitution(source, target, "p"), x)

    def test_infer_rejects_partial_inconsistent_and_multiple_changes(self):
        source = parse_formula("p -> (q -> p)")
        for text in ("r -> (q -> p)", "r -> (q -> s)", "a -> (b -> a)"):
            for variable in ("p", "q"):
                with self.subTest(text=text, variable=variable):
                    self.assertIsNone(infer_substitution(source, parse_formula(text), variable))

    def test_infer_rejects_changed_operations(self):
        for before, after in (("!p", "p"), ("p -> q", "!p"), ("!p", "p -> q")):
            with self.subTest(before=before, after=after):
                self.assertIsNone(infer_substitution(parse_formula(before), parse_formula(after), "p"))

    def test_generated_substitutions_and_roundtrip(self):
        small = [Var("p"), Var("q"), Not(Var("p")), Implies(Var("p"), Var("q"))]
        sources = small + [Not(x) for x in small] + [Implies(a, b) for a in small for b in small]
        for source in sources:
            for variable in sorted(variables(source)):
                for replacement in small:
                    with self.subTest(source=format_formula(source), variable=variable,
                                      replacement=format_formula(replacement)):
                        target = substitute(source, variable, replacement)
                        self.assertEqual(infer_substitution(source, target, variable), replacement)
                        self.assertEqual(parse_formula(format_formula(target)), target)


class ParserTests(unittest.TestCase):
    def test_all_latin_lowercase_letters(self):
        for char in "abcdefghijklmnopqrstuvwxyz":
            self.assertEqual(parse_formula(char), Var(char))

    def test_methodical_valid_example(self):
        self.assertEqual(
            parse_formula("(((a) → b)) → (f → ((p → (q))))"),
            parse_formula("(a -> b) -> (f -> (p -> q))"),
        )

    def test_redundant_parentheses_and_whitespace(self):
        for text in ("p -> q", "(p)->(q)", "((p -> q))", "\t p\n ->  q "):
            with self.subTest(text=text):
                self.assertEqual(parse_formula(text), Implies(Var("p"), Var("q")))

    def test_unicode_aliases(self):
        for arrow in ("->", "→", "⊃"):
            for negation in ("!", "¬"):
                self.assertEqual(parse_formula(f"{negation}p {arrow} q"), parse_formula("!p -> q"))

    def test_right_associativity(self):
        self.assertEqual(parse_formula("p -> q -> r"), parse_formula("p -> (q -> r)"))
        self.assertNotEqual(parse_formula("p -> q -> r"), parse_formula("(p -> q) -> r"))

    def test_negation_priority_and_repetition(self):
        self.assertEqual(parse_formula("!p -> q"), Implies(Not(Var("p")), Var("q")))
        self.assertEqual(parse_formula("!!p"), Not(Not(Var("p"))))
        self.assertEqual(parse_formula("!(p -> q)"), Not(Implies(Var("p"), Var("q"))))

    def test_invalid_inputs(self):
        invalid = (
            "", " ", "p -> (p -> p))", "mn -> (p -> p)", "q -> (p p)",
            "-> a -> b -> c", "a -> b -> c ->", "()", "!", "(p", "p)",
            ")p(", "p q", "pq", "p(q)", "(p)(q)", "p !q", "p -> -> q",
            "p -> ()", "(!)", "p - > q", "p => q", "P", "р", "p1", "1",
            "p & q", "p | q", "p ∧ q", "p ∨ q", "p <-> q", "[p]", "p\x00",
            "__import__('os')", "p -> !", "(p ->)", "p -> q garbage",
        )
        for text in invalid:
            with self.subTest(text=repr(text)):
                with self.assertRaises(FormulaSyntaxError):
                    parse_formula(text)

    def test_error_positions(self):
        for text, position in (("p ->", 4), ("p)", 1), ("p $ q", 2), ("pq", 1)):
            with self.subTest(text=text):
                with self.assertRaises(FormulaSyntaxError) as caught:
                    parse_formula(text)
                self.assertEqual(caught.exception.position, position)

    def test_length_limit(self):
        with self.assertRaisesRegex(FormulaSyntaxError, "длинный"):
            parse_formula("p" + " " * MAX_INPUT_LENGTH)

    def test_token_limit(self):
        with self.assertRaisesRegex(FormulaSyntaxError, "элементов"):
            tokenize("p " * (MAX_TOKENS + 1))

    def test_depth_limits(self):
        for text in (
            "!" * MAX_DEPTH + "p",
            "(" * MAX_PARSE_DEPTH + "p" + ")" * MAX_PARSE_DEPTH,
            "p -> " * MAX_DEPTH + "p",
        ):
            with self.subTest(length=len(text)):
                with self.assertRaises(FormulaSyntaxError):
                    parse_formula(text)
        allowed = parse_formula("!" * (MAX_DEPTH - 1) + "p")
        self.assertEqual(tree_depth(allowed), MAX_DEPTH)
        self.assertEqual(parse_formula(format_formula(allowed)), allowed)
        self.assertEqual({allowed: True}[allowed], True)

    def test_printed_deep_implication_can_be_read_back(self):
        # Полная расстановка скобок не должна нарушать допустимый ввод.
        formula = parse_formula("p -> " * (MAX_DEPTH - 1) + "p")
        self.assertEqual(tree_depth(formula), MAX_DEPTH)
        self.assertEqual(parse_formula(format_formula(formula)), formula)

    def test_deterministic_malformed_inputs_do_not_crash(self):
        generator = random.Random(2026)
        alphabet = "pq!()-> ¬⊃$1"
        for _ in range(1000):
            text = "".join(generator.choices(alphabet, k=generator.randrange(0, 45)))
            try:
                formula = parse_formula(text)
            except FormulaSyntaxError:
                continue
            self.assertEqual(parse_formula(format_formula(formula)), formula)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.session = ProofSession()

    def submit(self, text):
        return self.session.submit(parse_formula(text))

    def test_axioms_available_without_input(self):
        for axiom in AXIOMS:
            result = self.session.submit(axiom.formula)
            self.assertEqual(result.status, "known")
            self.assertEqual(result.entry.label, axiom.label)
        self.assertEqual(self.session.steps, ())

    def test_atomic_formula_and_axiom_subformulas_are_not_known(self):
        for text in ("p", "q", "q -> p", "p -> q"):
            self.assertEqual(self.submit(text).status, "rejected")
        self.assertEqual(self.session.steps, ())

    def test_tautology_requires_intermediate_steps(self):
        self.assertEqual(self.submit("p -> p").status, "rejected")

    def test_one_substitution(self):
        entry = self.submit("p -> (r -> p)").entry
        self.assertEqual((entry.label, entry.rule, entry.sources, entry.variable), ("1", "beta", ("A1",), "q"))
        self.assertEqual(entry.replacement, Var("r"))

    def test_no_simultaneous_substitution(self):
        self.assertEqual(self.submit("a -> (b -> a)").status, "rejected")
        self.assertEqual(self.submit("a -> (q -> a)").status, "accepted")
        self.assertEqual(self.submit("a -> (b -> a)").status, "accepted")
        self.assertEqual(len(self.session.steps), 2)

    def test_partial_or_inconsistent_substitution_rejected(self):
        for text in ("r -> (q -> p)", "r -> (q -> s)"):
            self.assertEqual(self.submit(text).status, "rejected")

    def test_compound_self_containing_replacement(self):
        result = self.submit("(p -> r) -> (q -> (p -> r))")
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.entry.replacement, parse_formula("p -> r"))

    def test_negation_from_third_axiom(self):
        result = self.submit("(!!r -> !q) -> (q -> !r)")
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.entry.sources, ("A3",))
        self.assertEqual(result.entry.variable, "p")
        result2 = self.submit("(!!r -> !(a -> b)) -> ((a -> b) -> !r)")
        self.assertEqual(result2.status, "accepted")
        self.assertEqual(result2.entry.sources, ("1",))

    def test_repeated_input_does_not_add_step(self):
        self.submit("p -> (r -> p)")
        result = self.submit("((p) ⊃ ((r) → (p)))")
        self.assertEqual(result.status, "known")
        self.assertEqual(result.entry.label, "1")
        self.assertEqual(len(self.session.steps), 1)

    def test_full_proof_rules_and_sources(self):
        expected = (
            ("beta", ("A2",)), ("beta", ("1",)), ("beta", ("2",)),
            ("beta", ("A1",)), ("mp", ("3", "4")),
            ("beta", ("5",)), ("mp", ("6", "A1")),
        )
        for index, (text, (rule, sources)) in enumerate(zip(PROOF, expected), 1):
            with self.subTest(step=index):
                result = self.submit(text)
                self.assertEqual(result.status, "accepted")
                self.assertEqual((result.entry.rule, result.entry.sources), (rule, sources))
                self.assertEqual(result.entry.label, str(index))

    def test_missing_mp_premise(self):
        for text in PROOF[:3]:
            self.submit(text)
        self.assertEqual(self.submit(PROOF[4]).status, "rejected")
        self.submit(PROOF[3])
        self.assertEqual(self.submit(PROOF[4]).entry.rule, "mp")

    def test_reverse_mp_is_not_permitted(self):
        # Получаем q -> A1, имеем A1; неизвестное q выводить нельзя.
        axiom = AXIOMS[0].formula
        first = substitute(axiom, "p", axiom)
        self.assertEqual(self.session.submit(first).status, "accepted")
        second = Implies(Var("q"), axiom)
        self.assertEqual(self.session.submit(second).entry.rule, "mp")
        self.assertEqual(self.submit("q").status, "rejected")

    def test_mp_requires_structural_not_semantic_match(self):
        axiom = AXIOMS[0].formula
        double_negation = Not(Not(axiom))
        source = substitute(axiom, "p", double_negation)
        self.assertEqual(self.session.submit(source).status, "accepted")
        # Есть A1, но нет !!A1. Заменять посылку по семантическому равенству нельзя.
        target = Implies(Var("q"), double_negation)
        self.assertEqual(self.session.submit(target).status, "rejected")

    def test_methodical_wrong_parentheses_are_rejected(self):
        for text in PROOF[:4]:
            self.submit(text)
        self.assertEqual(self.submit("p -> r -> (p -> p)").status, "rejected")
        self.assertEqual(self.submit(PROOF[4]).status, "accepted")

    def test_no_combined_beta_and_mp_in_one_step(self):
        for text in PROOF[:5]:
            self.submit(text)
        self.assertEqual(self.submit("p -> p").status, "rejected")
        self.submit(PROOF[5])
        self.assertEqual(self.submit("p -> p").status, "accepted")

    def test_rejected_formula_does_not_pollute_session(self):
        before = self.session.steps
        self.submit("p")
        self.assertEqual(self.session.steps, before)
        self.assertIsNone(self.session.known_entry(Var("p")))
        self.assertEqual(self.submit("q -> p").status, "rejected")
        self.assertEqual(self.submit("p -> (r -> p)").entry.label, "1")

    def test_goal_is_not_a_premise(self):
        self.session.set_goal(Var("p"))
        self.assertFalse(self.session.goal_reached)
        self.assertEqual(self.submit("p").status, "rejected")
        self.assertEqual(self.submit("q -> p").status, "rejected")

    def test_goal_reached_only_when_known(self):
        self.session.set_goal(parse_formula("p -> p"))
        for text in PROOF[:-1]:
            self.submit(text)
            self.assertFalse(self.session.goal_reached)
        self.submit(PROOF[-1])
        self.assertTrue(self.session.goal_reached)

    def test_axiom_can_be_goal(self):
        self.session.set_goal(AXIOMS[0].formula)
        self.assertTrue(self.session.goal_reached)
        self.assertEqual(self.session.steps, ())

    def test_reset_clears_goal_and_derived_formulas(self):
        for text in PROOF:
            self.submit(text)
        self.session.set_goal(parse_formula("p -> p"))
        self.session.reset()
        self.assertIsNone(self.session.goal)
        self.assertEqual(self.session.steps, ())
        self.assertEqual(self.submit("p -> p").status, "rejected")
        self.assertEqual(self.submit("p -> (r -> p)").entry.label, "1")

    def test_separate_sessions_are_independent(self):
        self.submit("p -> (r -> p)")
        self.assertEqual(ProofSession().steps, ())

    def test_accepted_formulas_are_tautologies(self):
        # Таблицы истинности здесь — независимая проверка корректности тестовых
        # выводов. В рабочем верификаторе семантической подмены правил нет.
        for text in PROOF:
            self.submit(text)
        for entry in (*AXIOMS, *self.session.steps):
            names = sorted(variables(entry.formula))
            for values in product((False, True), repeat=len(names)):
                with self.subTest(entry=entry.label, values=values):
                    self.assertTrue(evaluate(entry.formula, dict(zip(names, values))))


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.app = ConsoleApp()

    def process(self, text):
        output = StringIO()
        with redirect_stdout(output):
            result = self.app.process_line(text)
        return result, output.getvalue()

    def test_blank_input(self):
        result, output = self.process("   ")
        self.assertEqual(result, (True, False))
        self.assertEqual(output, "")

    def test_syntax_error_then_recovery(self):
        result, output = self.process("p ->")
        self.assertEqual(result, (False, False))
        self.assertIn("ОШИБКА ЗАПИСИ", output)
        result, output = self.process("p -> (r -> p)")
        self.assertEqual(result, (True, False))
        self.assertIn("ПРИНЯТО — шаг 1", output)

    def test_rejected_formula_explanation(self):
        result, output = self.process("p -> p")
        self.assertEqual(result, (False, False))
        self.assertIn("Запись корректна", output)
        self.assertIn("не означает ложность", output)

    def test_parse_only_does_not_prove(self):
        result, output = self.process("/parse p -> p")
        self.assertEqual(result, (True, False))
        self.assertIn("ЗАПИСЬ КОРРЕКТНА", output)
        self.assertEqual(self.app.session.steps, ())
        self.assertEqual(self.process("p -> p")[0], (False, False))

    def test_invalid_commands(self):
        for text in ("/", "/unknown", "/goal", "/parse", "/exit extra", "/list x", "/reset x", "/help x"):
            with self.subTest(text=text):
                self.assertEqual(self.process(text)[0], (False, False))

    def test_invalid_goal_preserves_previous_goal(self):
        self.process("/goal p -> p")
        result, output = self.process("/goal p ->")
        self.assertEqual(result, (False, False))
        self.assertIn("аргументе команды", output)
        self.assertEqual(self.app.session.goal, parse_formula("p -> p"))

    def test_goal_and_full_proof_output(self):
        self.process("/goal p -> p")
        for text in PROOF:
            result, output = self.process(text)
            self.assertEqual(result, (True, False))
        self.assertIn("ЦЕЛЬ ДОСТИГНУТА", output)
        self.assertIn("A -> B [6]", output)
        self.assertIn("A      [A1]", output)
        self.assertIn("modus ponens", output)

    def test_beta_output_contains_actual_replacement_and_source(self):
        _, output = self.process("!r -> (q -> !r)")
        self.assertIn("Источник [A1]", output)
        self.assertIn("Замена: p на !r", output)

    def test_axiom_and_duplicate_output(self):
        _, output = self.process("p -> (q -> p)")
        self.assertIn("исходная аксиома A1", output)
        self.process("p -> (r -> p)")
        _, output = self.process("p -> (r -> p)")
        self.assertIn("получена на шаге 1", output)

    def test_help_list_reset_exit(self):
        self.assertIn("/parse", self.process("/help")[1])
        self.assertIn("Пока нет", self.process("/list")[1])
        self.process("p -> (r -> p)")
        self.assertIn("подстановка из A1", self.process("/list")[1])
        self.process("/goal p -> p")
        self.process("/reset")
        self.assertEqual(self.app.session.steps, ())
        self.assertIsNone(self.app.session.goal)
        self.assertEqual(self.process("/exit")[0], (True, True))

    def test_interactive_eof(self):
        with patch("builtins.input", side_effect=EOFError), redirect_stdout(StringIO()):
            self.assertEqual(self.app.interactive(), 0)

    def test_interactive_session(self):
        lines = ["p ->", "p -> (r -> p)", "/list", "/exit"]
        output = StringIO()
        with patch("builtins.input", side_effect=lines), redirect_stdout(output):
            self.assertEqual(self.app.interactive(), 0)
        self.assertIn("ПРИНЯТО", output.getvalue())
        self.assertIn("Работа завершена", output.getvalue())


class FileAndProcessTests(unittest.TestCase):
    def run_file(self, content=None, raw=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.txt"
            if raw is not None:
                path.write_bytes(raw)
            elif content is not None:
                path.write_text(content, encoding="utf-8")
            app = ConsoleApp()
            output = StringIO()
            with redirect_stdout(output):
                code = app.run_file(path)
            return code, output.getvalue(), app

    def test_example_files(self):
        expected = {
            "proof_p_implies_p.txt": (0, 7),
            "negation.txt": (0, 2),
            "atomic_substitution.txt": (0, 2),
            "errors.txt": (1, 1),
        }
        for filename, (code, steps) in expected.items():
            with self.subTest(filename=filename), redirect_stdout(StringIO()):
                app = ConsoleApp()
                self.assertEqual(app.run_file(ROOT / "examples" / filename), code)
                self.assertEqual(len(app.session.steps), steps)

    def test_utf8_bom_and_crlf(self):
        raw = b"\xef\xbb\xbf# comment\r\np -> (r -> p)\r\n"
        self.assertEqual(self.run_file(raw=raw)[0], 0)

    def test_empty_or_comment_only_file(self):
        for text in ("", "\n# comment\n", "/help\n"):
            self.assertEqual(self.run_file(content=text)[0], 1)

    def test_invalid_encoding(self):
        self.assertEqual(self.run_file(raw=b"\xff\xfe")[0], 2)

    def test_missing_file(self):
        self.assertEqual(self.run_file()[0], 2)

    def test_oversized_file(self):
        self.assertEqual(self.run_file(raw=b" " * 1_048_577)[0], 2)

    def test_unreached_goal_makes_batch_fail(self):
        self.assertEqual(self.run_file(content="/goal p -> p\np -> (r -> p)\n")[0], 1)

    def test_batch_continues_after_error_but_remains_failed(self):
        code, output, app = self.run_file(content="p ->\np -> (r -> p)\n")
        self.assertEqual(code, 1)
        self.assertEqual(len(app.session.steps), 1)
        self.assertIn("ОШИБКА ЗАПИСИ", output)

    def test_demo_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), "--demo"],
                cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ЦЕЛЬ ДОСТИГНУТА", result.stdout)
        self.assertIn("Принятых шагов: 7", result.stdout)

    def test_subprocess_redirected_input(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py")],
            input="/parse ¬p ⊃ q\n/exit\n", capture_output=True,
            text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ЗАПИСЬ КОРРЕКТНА", result.stdout)

    def test_mutually_exclusive_cli_modes(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--demo", "--file", "x"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
