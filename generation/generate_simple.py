import re
import pandas as pd
import random

def equation(args: list) -> str:
    if len(args) == 1:
        return (str(args[0]) if args[0] != 0 else "") + " + C"
    if len(args) == 2:
        return (
            (str(args[0]) if args[0] != 0 else "")
            + (f"+{args[1]}x" if args[1] != 0 else "")
            + "+ C"
        )
    return (
        re.sub(
            r"\++",
            "+",
            (str(args[0]) if args[0] != 0 else "")
            + (f"+{args[1]}x" if args[1] != 0 else "")
            + "+"
            + "+".join(
                [
                    str(coef) + "x^{" + str(deg + 2) + "}" if coef != 0 else ""
                    for deg, coef in enumerate(args[2:])
                ]
            ),
        )
        + " + C"
    )


def deriv(args: list) -> str:
    # b + a * x -> a
    if len(args) == 2:
        return str(args[1])
    # c + x * b + x^2 * a -> 2 * c * x + b
    if len(args) == 3:
        return (str(args[1]) if args[1] != 0 else "") + f"+{2 * args[2]}x"
    return re.sub(
        r"\++",
        "+",
        (str(args[1]) if args[1] != 0 else "")
        + (f"+{2 * args[2]}x" if args[2] != 0 else "")
        + "+"
        + "+".join(
            [
                str(coef * (deg + 3)) + "x^{" + str(deg + 2) + "}" if coef != 0 else ""
                for deg, coef in enumerate(args[3:])
            ]
        ),
    )

def generate(d: dict = None) -> pd.DataFrame:
    if d is None:
        d = {i : random.randint(10, 100) for i in range(2, 10)}
    equations = []
    answers = []
    for key, val in d.items():
        for _ in range(val):
            args = random.sample(range(-10, 10), key)
            equations.append('y^{\prime}=' + re.sub(r'\+-', '-', deriv(args)))
            answers.append('y=' + re.sub(r'\+-', '-', equation(args)))
            # print(f"Answer for equation {equations[-1]} is {answers[-1]}")
    return pd.DataFrame({"equation": equations, "answer": answers})

