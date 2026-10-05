"""Лексический анализ и рекурсивный спуск без eval и сторонних библиотек.

Грамматика:
    implication := unary [ IMPLIES implication ]
    unary       := NOT unary | atom
    atom        := VARIABLE | '(' implication ')'

Импликация правоассоциативна; отрицание имеет более высокий приоритет.
"""

from dataclasses import dataclass

from formula import Formula, Implies, Not, Var, tree_depth


MAX_INPUT_LENGTH = 10_000
MAX_TOKENS = 2_000
MAX_DEPTH = 100
MAX_PARSE_DEPTH = 200


class FormulaSyntaxError(ValueError):
    """Ожидаемая ошибка пользовательского ввода, position отсчитывается от 0."""

    def __init__(self, message: str, position: int):
        super().__init__(message)
        self.message = message
        self.position = position


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    text: str
    position: int


def tokenize(text: str) -> list[Token]:
    if len(text) > MAX_INPUT_LENGTH:
        raise FormulaSyntaxError(
            f"Слишком длинный ввод: не более {MAX_INPUT_LENGTH} символов.",
            MAX_INPUT_LENGTH,
        )
    tokens: list[Token] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            i += 1
            continue
        start = i
        if text.startswith("->", i):
            kind, value = "IMPLIES", "->"
            i += 2
        elif char in "→⊃":
            kind, value = "IMPLIES", char
            i += 1
        elif char in "!¬":
            kind, value = "NOT", char
            i += 1
        elif char in "()":
            kind, value = ("LPAREN" if char == "(" else "RPAREN"), char
            i += 1
        elif "a" <= char <= "z":
            kind, value = "VAR", char
            i += 1
        else:
            raise FormulaSyntaxError(
                f"Недопустимый символ {char!r}. Разрешены a-z, ->, →, ⊃, !, ¬ и скобки.",
                i,
            )
        tokens.append(Token(kind, value, start))
        if len(tokens) > MAX_TOKENS:
            raise FormulaSyntaxError(
                f"Слишком много элементов: не более {MAX_TOKENS}.", start
            )
    tokens.append(Token("EOF", "", len(text)))
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def check_depth(self, depth: int) -> None:
        if depth >= MAX_PARSE_DEPTH:
            raise FormulaSyntaxError(
                f"Слишком глубокая синтаксическая вложенность. Предел: {MAX_PARSE_DEPTH} уровней.",
                self.current.position,
            )

    def parse(self) -> Formula:
        if self.current.kind == "EOF":
            raise FormulaSyntaxError("Пустая строка не является формулой.", 0)
        result = self.parse_implication(0)
        if self.current.kind != "EOF":
            token = self.current
            if token.kind == "RPAREN":
                message = "Лишняя закрывающая скобка."
            else:
                message = (
                    "Между выражениями пропущена операция. "
                    "Имя переменной должно состоять из одной буквы."
                )
            raise FormulaSyntaxError(message, token.position)
        if tree_depth(result) > MAX_DEPTH:
            raise FormulaSyntaxError(
                f"Дерево формулы глубже {MAX_DEPTH} уровней.", 0
            )
        return result

    def parse_implication(self, depth: int) -> Formula:
        self.check_depth(depth)
        left = self.parse_unary(depth)
        if self.current.kind == "IMPLIES":
            self.advance()
            # Рекурсия справа: p -> q -> r означает p -> (q -> r).
            return Implies(left, self.parse_implication(depth + 1))
        return left

    def parse_unary(self, depth: int) -> Formula:
        self.check_depth(depth)
        if self.current.kind == "NOT":
            self.advance()
            return Not(self.parse_unary(depth + 1))
        return self.parse_atom(depth)

    def parse_atom(self, depth: int) -> Formula:
        token = self.current
        if token.kind == "VAR":
            self.advance()
            return Var(token.text)
        if token.kind == "LPAREN":
            opening = self.advance()
            if self.current.kind == "RPAREN":
                raise FormulaSyntaxError("Пустые скобки недопустимы.", self.current.position)
            result = self.parse_implication(depth + 1)
            if self.current.kind != "RPAREN":
                if self.current.kind == "EOF":
                    message = (
                        "Не хватает закрывающей скобки для открывающей "
                        f"в позиции {opening.position + 1}."
                    )
                else:
                    message = "Внутри скобок пропущена операция или закрывающая скобка."
                raise FormulaSyntaxError(message, self.current.position)
            self.advance()
            return result
        if token.kind == "IMPLIES":
            message = "Перед импликацией ожидалась формула: переменная, отрицание или скобки."
        elif token.kind == "RPAREN":
            message = "Перед закрывающей скобкой ожидалась формула."
        else:
            message = "Строка закончилась, но ожидалась формула после операции или скобки."
        raise FormulaSyntaxError(message, token.position)


def parse_formula(text: str) -> Formula:
    """Единая точка входа: возвращает дерево либо FormulaSyntaxError."""
    parser = Parser(tokenize(text))
    try:
        return parser.parse()
    except RecursionError as error:
        # Дополнительная защита, если окружение уменьшило предел рекурсии Python.
        raise FormulaSyntaxError(
            "Слишком глубокая формула для текущего окружения Python.",
            parser.current.position,
        ) from error
