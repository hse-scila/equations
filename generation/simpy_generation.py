from sympy import latex, symbols, diff, parse_expr
import random
import pandas as pd
from itertools import combinations, permutations

base_funcs = [
    "x",
    "x^{2}",
    "x^{3}",
    "\sin(x)",
    "\sin(2 \cdot x)",
    "\cos(x)",
    "\cos(2 \cdot x)",
    "\\tg(x)",
    "\\tg(2 \cdot x)",
    "\ctg(x)",
    "\ctg(2 \cdot x)",
    "\\arcsin(x)",
    "\\arcsin(2 \cdot x)",
    "\\arccos(x)",
    "\\arccos(2 \cdot x)",
    "\\arctg(x)",
    "\\arctg(2 \cdot x)",
    "\\arcctg(x)",
    "\\arcctg(2 \cdot x)",
    "\\ln(x)",
    "\\ln(2 \cdot x)",
    "e^{x}",
    "e^{2 \cdot x}",
    # "1",
]

derives = [
    "1",
    "2x",
    "3x^{2}",
    "\cos(x)",
    "2\cos(2 \cdot x)",
    "-\sin(x)",
    "-2\sin(2 \cdot x)",
    "\\frac{1}{\cos^{2}(x)}",
    "\\frac{2}{\cos^{2}(2 \cdot x)}",
    "\\frac{1}{\sin^{2}(x)}",
    "\\frac{2}{\sin^{2}(2 \cdot x)}",
    "\\frac{1}{\sqrt{1 - x^{2}}}",
    "\\frac{2}{\sqrt{1 - 4 \cdot x^{2}}}",
    "-\\frac{1}{\sqrt{1 - x^{2}}}",
    "-\\frac{2}{\sqrt{1 - 4 \cdot x^{2}}}",
    "\\frac{1}{x^{2} + 1}",
    "\\frac{2}{4x^{2} + 1}",
    "-\\frac{1}{x^{2} + 1}",
    "-\\frac{2}{4x^{2} + 1}",
    "\\frac{1}{x}",
    "\\frac{2}{x}",
    "e^{x}",
    "2 \cdot e^{2 \cdot x}",
    # "0"
]

def simplest_base():
    for func, deriv in zip(base_funcs, derives):
        for i in range(-100000000, 1000000000):
            if i == 0:
                continue
            yield "y = " + f"{i} \cdot {func}"+" + C", "y^{\prime} = " + f"{i} \cdot {deriv}"


def parse_frac(frac):
    if "\\frac" not in frac:
        return frac
    result = ''
    while "\\frac" in frac:
        idx = frac.find('\\frac')
        result = frac[:idx]
        numerator_start = idx + 6
        numerator_end = numerator_start + 1
        balance = 1
        while balance > 0:
            if frac[numerator_end] == '{':
                balance += 1
            elif frac[numerator_end] == '}':
                balance -= 1
            numerator_end += 1
        numerator = frac[numerator_start: numerator_end - 1]
        result += f"({numerator}) / ("
        denominator_start = numerator_end + 1
        denominator_end = denominator_start + 1
        balance = 1
        while balance > 0:
            if frac[denominator_end] == '{':
                balance += 1
            elif frac[denominator_end] == '}':
                balance -= 1
            denominator_end += 1
       
        denominator = frac[denominator_start: denominator_end - 1]
        result += f"{denominator})"

        frac = result + frac[denominator_end:]
    return f'({result})'

def tex2sympy(text):
    text = parse_frac(text)
    return (
        text.replace("{", "(")
        .replace("}", ")")
        .replace("^", "**")
        .replace("\cdot", "*")
        .replace("\\", "")
        .replace("arctg", "atan")
        .replace("arcctg", "acot")
        .replace("arcsin", "asin")
        .replace("arccos", "acos")
        .replace("tg", "tan")
        .replace("ctg", "cot")
    )

def calc_derive(latex_function):
    latex_function = tex2sympy(latex_function)
    function_expr = parse_expr(latex_function)
    x = symbols("x")
    derivative_expr = diff(function_expr, x)
    return latex(derivative_expr)


def level1(term_count=1):
    """
    returns set of functions and it's derivaties of the form
    y = \sum_{i=0}^{term_count}f(x)
    where f(x) is a base_func
    """
    for subset in combinations(base_funcs, term_count):
        for perm in permutations(subset):
            latex_func = perm[0]
            for func in perm[1:]:
                op = random.choice([" + ", " - "])
                latex_func += op + func
            yield "y=" + latex_func + "+C", "y^{\prime}=" + calc_derive(latex_func)

def create_terms(complexity):
    for subset in combinations(base_funcs, complexity):
        for perm in permutations(subset):
            latex_func = perm[0]
            for func in perm[1:]:
                op = random.choice(["*", "/"])
                if op == "*":
                    latex_func += ' \cdot ' + func
                else:
                    latex_func += ' \cdot \\frac{1}{' + func + '}'
            yield latex_func


def level2(term_count=2, complexity=2):
    """
    returns set of functions and it's derivaties of the form
    y = \sum_{i=0}^{term_count}f(x)
    where f(x) is a product or fraction of base_funcs
    """
    for subset in combinations(create_terms(complexity), term_count):
        print(subset)
        for perm in permutations(subset):
            latex_func = perm[0]
            for func in perm[1:]:
                op = random.choice(["+", "-"])
                latex_func += op + func
            try:
                yield "y=" + latex_func + "+C", "y^{\prime}=" + calc_derive(latex_func)
            except:
                print(latex_func)

#TODO: write a function 'level3' that will create more complicated funcions
# of the form (f(x) + g(x)) / (p(x) + q(x)) and other

if __name__ == "__main__":
    pd.DataFrame(level2()).to_csv('divided_vars.csv', index=False)
