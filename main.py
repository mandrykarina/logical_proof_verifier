"""Консольный интерфейс. Запуск: python main.py; демонстрация: --demo."""

import argparse
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Требуется Python 3.10 или новее. Проверьте: python --version")

from formula import format_formula, tree_depth, variables
from parser import FormulaSyntaxError, parse_formula
from verifier import AXIOMS, Entry, ProofSession


HELP = """Команды:
  /help             Справка.
  /list             Аксиомы и принятые шаги с обоснованиями.
  /goal ФОРМУЛА     Установить цель; цель не становится посылкой.
  /parse ФОРМУЛА    Проверить только запись, не добавляя её в доказательство.
  /reset            Очистить шаги и цель; оставить исходные аксиомы.
  /exit             Завершить работу.

Переменные: одна маленькая латинская буква a-z.
Импликация: ->, → или ⊃. Отрицание: ! или ¬.
Отрицание имеет приоритет: !p -> q означает (!p) -> q.
Без скобок p -> q -> r означает p -> (q -> r).
Одна строка = один MP или одна подстановка одной переменной.
Пустые строки пропускаются. В файлах строки с # — комментарии.
"""


def rule_description(entry: Entry) -> str:
    if entry.rule == "axiom":
        return "исходная аксиома"
    if entry.rule == "mp":
        return f"MP из {entry.sources[0]} и {entry.sources[1]}"
    assert entry.replacement is not None
    return (
        f"подстановка из {entry.sources[0]}: "
        f"{entry.variable} := {format_formula(entry.replacement)}"
    )


class ConsoleApp:
    def __init__(self) -> None:
        self.session = ProofSession()
        self.formulas_entered = 0

    def show_intro(self) -> None:
        print("Проверка логических доказательств — вариант 2")
        print("Операции: -> (импликация), ! (отрицание). Переменные: a-z.")
        print("Аксиомы известны с самого начала:")
        for entry in AXIOMS:
            print(f"  {entry.label}: {format_formula(entry.formula)}")
        print("Вводите формулы по одной. Справка: /help\n")

    def show_history(self) -> None:
        print("\nИСХОДНЫЕ АКСИОМЫ")
        for entry in AXIOMS:
            print(f"  {entry.label}: {format_formula(entry.formula)}")
        print("ПРИНЯТЫЕ ШАГИ")
        if not self.session.steps:
            print("  Пока нет принятых шагов.")
        for entry in self.session.steps:
            print(f"  {entry.label}. {format_formula(entry.formula)}")
            print(f"     {rule_description(entry)}")
        if self.session.goal is not None:
            print(f"Цель: {format_formula(self.session.goal)}")
            print("Статус: доказана." if self.session.goal_reached else "Статус: пока не доказана.")

    def report_goal(self) -> None:
        if self.session.goal_reached:
            assert self.session.goal is not None
            entry = self.session.known_entry(self.session.goal)
            assert entry is not None
            print(
                f"ЦЕЛЬ ДОСТИГНУТА: {format_formula(self.session.goal)} "
                f"[основание: {entry.label}]."
            )

    def process_line(self, line: str) -> tuple[bool, bool]:
        """Возвращает (строка обработана успешно, нужно завершить работу)."""
        if not line.strip():
            return True, False
        try:
            if line.lstrip().startswith("/"):
                return self.process_command(line.strip())
            self.formulas_entered += 1
            formula = parse_formula(line)
            result = self.session.submit(formula)
        except FormulaSyntaxError as error:
            print(f"ОШИБКА ЗАПИСИ, позиция {error.position + 1}: {error.message}")
            print("Состояние доказательства не изменено.")
            return False, False

        if result.status == "rejected":
            print("ШАГ НЕ ПРИНЯТ")
            print(f"Формула: {format_formula(formula)}")
            print("Запись корректна, но обоснование одним MP или одной подстановкой не найдено.")
            print("Это не означает ложность или недоказуемость формулы.")
            print("Формула не добавлена. Проверьте промежуточные шаги и скобки.")
            return False, False

        entry = result.entry
        assert entry is not None
        if result.status == "known":
            if entry.rule == "axiom":
                print(f"УЖЕ ИЗВЕСТНО: это исходная аксиома {entry.label}.")
            else:
                print(f"УЖЕ ИЗВЕСТНО: формула получена на шаге {entry.label}.")
            print(f"Формула: {format_formula(formula)}")
            print("Повторно в список шагов не добавляется.")
        else:
            print(f"ПРИНЯТО — шаг {entry.label}")
            print(f"Формула: {format_formula(formula)}")
            if entry.rule == "mp":
                implication = self.session.get_entry(entry.sources[0])
                premise = self.session.get_entry(entry.sources[1])
                print("Правило: modus ponens (MP).")
                print(f"  A -> B [{implication.label}]: {format_formula(implication.formula)}")
                print(f"  A      [{premise.label}]: {format_formula(premise.formula)}")
                print(f"  Следовательно, B: {format_formula(formula)}")
            else:
                source = self.session.get_entry(entry.sources[0])
                assert entry.replacement is not None
                print("Правило: бета (подстановка).")
                print(f"  Источник [{source.label}]: {format_formula(source.formula)}")
                print(f"  Замена: {entry.variable} на {format_formula(entry.replacement)}")
                print("  Заменены все вхождения одной переменной; остальные части сохранены.")
            print("Формула добавлена в доказательство.")
        self.report_goal()
        return True, False

    def process_command(self, text: str) -> tuple[bool, bool]:
        parts = text.split(maxsplit=1)
        command = parts[0]
        argument = parts[1] if len(parts) == 2 else ""
        if command in {"/goal", "/parse"}:
            if not argument:
                print(f"ОШИБКА КОМАНДЫ: укажите формулу после {command}.")
                return False, False
            # Сначала полный разбор, затем изменение цели: ошибка не портит состояние.
            try:
                formula = parse_formula(argument)
            except FormulaSyntaxError as error:
                print(f"ОШИБКА ЗАПИСИ в аргументе команды, позиция {error.position + 1}: {error.message}")
                print("Состояние доказательства не изменено.")
                return False, False
            if command == "/goal":
                self.session.set_goal(formula)
                print(f"Цель: {format_formula(formula)}")
                print("Цель не добавляется в известные формулы.")
                self.report_goal()
            else:
                print("ЗАПИСЬ КОРРЕКТНА")
                print(f"Формула: {format_formula(formula)}")
                print(f"Переменные: {', '.join(sorted(variables(formula)))}")
                print(f"Глубина дерева: {tree_depth(formula)}")
                print("Проверен только синтаксис. Формула не добавлена в доказательство.")
            return True, False

        if command not in {"/help", "/list", "/reset", "/exit"}:
            print(f"ОШИБКА КОМАНДЫ: неизвестная команда {command!r}. Справка: /help.")
            return False, False
        if argument:
            print(f"ОШИБКА КОМАНДЫ: {command} не принимает аргументы.")
            return False, False
        if command == "/help":
            print(HELP)
        elif command == "/list":
            self.show_history()
        elif command == "/reset":
            self.session.reset()
            print("Шаги и цель очищены. Три исходные аксиомы доступны.")
        else:
            print("Работа завершена.")
            return True, True
        return True, False

    def interactive(self) -> int:
        self.show_intro()
        while True:
            try:
                line = input("\nФормула > ")
            except EOFError:
                print("\nКонец ввода. Работа завершена.")
                return 0
            _, should_exit = self.process_line(line)
            if should_exit:
                return 0

    def run_file(self, path: Path) -> int:
        try:
            if path.stat().st_size > 1_048_576:
                print("ОШИБКА ФАЙЛА: предел размера — 1 МиБ.")
                return 2
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            print(f"ОШИБКА ФАЙЛА: {error}")
            print("Проверьте путь и кодировку UTF-8.")
            return 2

        self.show_intro()
        errors = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            preview = line if len(line) <= 200 else line[:200] + " ... [строка сокращена в выводе]"
            print(f"\nСтрока файла {line_number} > {preview}")
            success, should_exit = self.process_line(line)
            if not success:
                errors += 1
            if should_exit:
                break
        self.show_history()
        if self.formulas_entered == 0:
            print("ОШИБКА: файл не содержит введённых формул доказательства.")
            errors += 1
        goal_missing = self.session.goal is not None and not self.session.goal_reached
        print(f"\nОшибок или отклонённых строк: {errors}. Принятых шагов: {len(self.session.steps)}.")
        if goal_missing:
            print("Указанная цель не достигнута.")
        return 1 if errors or goal_missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Верификатор доказательств. ЛР 1, вариант 2.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="показать доказательство p -> p")
    modes.add_argument("--file", type=Path, help="проверить доказательство из файла UTF-8")
    args = parser.parse_args(argv)
    app = ConsoleApp()
    if args.demo:
        return app.run_file(Path(__file__).resolve().parent / "examples" / "proof_p_implies_p.txt")
    if args.file is not None:
        return app.run_file(args.file)
    return app.interactive()


if __name__ == "__main__":
    # В том числе корректная запись русского вывода в файл через >.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nРабота прервана пользователем.")
        raise SystemExit(130)
    except UnicodeError:
        print("Ошибка кодировки. Используйте терминал и входные файлы в UTF-8.")
        raise SystemExit(2)
