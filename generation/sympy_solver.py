from sympy import *
import pandas as pd

x = symbols("x")
func2tex = {
    sin(x): "\sin(x)",
    sin(2 * x): "\sin(2 * x)",
    cos(x): "\cos(x)",
    cos(2 * x): "\cos(2 * x)",
    tan(x): "\\tg(x)",
    tan(2 * x): "\\tg(2x)",
    cot(x): "\\ctg(x)",
    cot(2 * x): "\\ctg(2x)",
    exp(x): "e^{x}",
    exp(2 * x): "e^{2 * x}",
    ln(x): "\ln(x)",
    ln(2 * x): "ln(2 * x)",
    0: "0",
    x: "x",
    2 * x: "2x",
    x**2: "x^{2}",
    x**3: "x^{3}",
    sqrt(x): "\sqrt(x)",
    sqrt(2 * x): "\sqrt(x)",
    asin(x): "\\arcsin(x)",
    asin(2 * x): "\\arcsin(2x)",
    acos(x): "\\arccos(x)",
    acos(2 * x): "\\arccos(2x)",
    atan(x): "\\arctg(x)",
    atan(2 * x): "arctg(2x)",
    acot(x): "arcctg(x)",
    acot(2 * x): "arcctg(2x)",
}


# TODO: make work
def simplyfy(text):
    return (
        text.replace("+ -", "-")
        .replace(" 0y^{\prime}", "")
        .replace(" 0y", "")
        .replace(" 1y", "y")
        .replace(" 1y^{\prime}", " y^{\prime}")
        .replace(" 1y^{\prime\prime}", " y^{\prime\prime}")
        .replace(" 0y^{\prime\prime}", "")
    )


def generate_ingomoghenous_second_order(start=-10, end=11, return_df=False):
    equations = []
    answers = []
    for a in range(start, end):
        if a == 0:
            continue
        for b in range(start, end):
            for c in range(start, end):
                for sp, tex in func2tex.items():
                    equations.append(
                        a
                        + "y^{\prime\prime} + "
                        + str(b)
                        + "y^{\prime} + "
                        + str(c)
                        + "y = "
                        + tex
                    )
                    y = Function("y")
                    equation = Eq(a * y(x).diff(x, x) + b * y(x).diff(x) + c * y(x), sp)
                    answers.append(latex(dsolve(equation)))
    if return_df:
        return pd.DataFrame({"equation": equations, "answer": answers})
    return (equations, answers)


def generate_ingomoghenous_third_order(start=-10, end=11, return_df=False):
    equations = []
    answers = []
    for a in range(start, end):
        if a == 0:
            continue
        for b in range(start, end):
            for c in range(start, end):
                for d in range(start, end):
                    for sp, tex in func2tex.items():
                        equations.append(
                            a
                            + "y^{\prime\prime\prime} + "
                            + str(b)
                            + "y^{\prime\prime} + "
                            + str(c)
                            + "y^{\prime}"
                            + str(d)
                            + "y = "
                            + tex
                        )
                        y = Function("y")
                        equation = Eq(
                            a * y(x).diff(x, x, x)
                            + b * y(x).diff(x, x)
                            + c * y(x).diff(x),
                            sp,
                        )
                    answers.append(latex(dsolve(equation)))
    if return_df:
        return pd.DataFrame({"equation": equations, "answer": answers})
    return (equations, answers)
