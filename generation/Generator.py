import pandas as pd
from sympy import *
from itertools import combinations, permutations
import random
import re
import concurrent.futures


class Generator:

    def __init__(self) -> None:
        self.x = symbols("x")
        self.func2tex = {
            sin(self.x): "\sin(x)",
            sin(2 * self.x): "\sin(2 * x)",
            cos(self.x): "\cos(x)",
            cos(2 * self.x): "\cos(2 * x)",
            # tan(self.x): "\\tg(x)",
            # tan(2 * self.x): "\\tg(2x)",
            # cot(self.x): "\\ctg(x)",
            # cot(2 * self.x): "\\ctg(2x)",
            exp(self.x): "e^{x}",
            exp(2 * self.x): "e^{2 * x}",
            # ln(self.x): "\ln(x)",
            # ln(2 * self.x): "ln(2 * x)",
            # 0: "0",
            self.x: "x",
            2 * self.x: "2x",
            self.x**2: "x^{2}",
            self.x**3: "x^{3}",
            # sqrt(self.x): "\sqrt(x)",
            # sqrt(2 * self.x): "\sqrt(x)",
            # asin(self.x): "\\arcsin(x)",
            # asin(2 * self.x): "\\arcsin(2x)",
            # acos(self.x): "\\arccos(x)",
            # acos(2 * self.x): "\\arccos(2x)",
            # atan(self.x): "\\arctg(x)",
            # atan(2 * self.x): "arctg(2x)",
            # acot(self.x): "arcctg(x)",
            # acot(2 * self.x): "arcctg(2x)",
            sinh(self.x): "sh(x)",
            sinh(2 * self.x): "sh(2x)",
            cosh(self.x): "ch(x)",
            cosh(2 * self.x): "ch(2x)"
        }

    @staticmethod
    def second_order_equation(x_1: int | float, x_2: int | float) -> str:
        """
        C_1 * e^x1 + C_2 * e^x2
        is solution for
        y^{\prime\prime} - (x1 + x2)y^{\prime} + x1 * x2 * y = 0
        """
        a = -x_1 - x_2
        a_component: str = ""
        prime1 = "{\prime}"
        if a > 0:
            a_component = f"+{a}y^{prime1}"
        elif a < 0:
            a_component = f"{a}y^{prime1}"
        b = x_1 * x_2
        b_component: str = ""
        if b > 0:
            b_component = f"+{b}y"
        elif b < 0:
            b_component = f"{b}y"
        prime2 = "{\prime\prime}"
        return f"y^{prime2}" + a_component + b_component + "=0"

    @staticmethod
    def fast_linear_homogeneous_second_order(
        a_range: list  = range(-10, 11),
        b_range: list  = range(-10, 11),
        return_df: bool = False,
    ) -> pd.DataFrame | tuple[list[str], list[str]]:
        """
        Generates second order homogeneous linear equations
        y^{\prime\prime} + ay^{\prime} + by = 0
        for all (a, b) \in [end - start] x [end - start]
        Except of a = b
        """
        equations = []
        answers = []
        for x1 in a_range:
            for x2 in b_range:
                if x1 == x2:
                    continue
                equations.append(Generator.second_order_equation(x1, x2))
                x1_component: str = "C_{1}"
                x2_component: str = "C_{2}"
                if x1 != 0:
                    x1_component += f"e^{'{' + str(x1) + 'x}'}"
                if x2 != 0:
                    x2_component += f"e^{'{' + str(x2) + 'x}'}"
                answers.append(x1_component + "+" + x2_component)
        if return_df:
            return pd.DataFrame({"equation": equations, "answer": answers})
        return (equations, answers)

    @staticmethod
    def third_order_equation(
        x_1: int | float, x_2: int | float, x_3: int | float
    ) -> str:
        """
        C_1 * e^x1 + C_2 * e^x2 + C_3 * e^x3
        is solution for
        y^{\prime\prime\prime} + ay^{\prime\prime} + by^{\prime} + cy = 0
        """
        a = -x_1 - x_2 - x_3
        b = x_1 * x_2 + x_2 * x_3 + x_1 * x_3
        c = - x_1 * x_2 * x_3
        prime1 = "{\prime}"
        prime2 = "{\prime\prime}"
        prime3 = "{\prime\prime\prime}"
        a_component: str = ""
        if a > 0:
            a_component = f"+{a}y^{prime2}"
        elif a < 0:
            a_component = f"{a}y^{prime2}"
        b_component: str = ""
        if b > 0:
            b_component = f"+{b}y^{prime1}"
        elif b < 0:
            b_component = f"{b}y^{prime1}"
        c_component: str = ""
        if c > 0:
            c_component = f"+{c}y"
        elif c < 0:
            c_component = f"{c}y"
        return f"y^{prime3}" + a_component + b_component + c_component + "=0"

    @staticmethod
    def fast_linear_homogeneous_third_order(
        a_range: list  = range(-10, 11),
        b_range: list  = range(-10, 11),
        c_range: list  = range(-10, 11),
        return_df: bool = False,
    ) -> pd.DataFrame | tuple[list[str], list[str]]:
        """
        Generates third order homogeneous linear equations
        y^{\prime\prime\prime} + ay^{\prime\prime} + by^{\prime} + cy = 0
        for all (a, b, c) \in [start, end - 1] x [start, end - 1] x [start, end - 1]
        Except of a = b or b = c or a = c
        (I was to lasy to hardcode these cases)
        """
        equations = []
        answers = []
        for x1 in a_range:
            for x2 in b_range:
                for x3 in c_range:
                    if x1 == x2 or x2 == x3 or x1 == x3:
                        continue
                    equations.append(Generator.third_order_equation(x1, x2, x3))
                    x1_component: str = "C_{1}"
                    x2_component: str = "C_{2}"
                    x3_component: str = "C_{3}"
                    if x1 != 0:
                        x1_component += f"e^{'{' + str(x1) + 'x}'}"
                    if x2 != 0:
                        x2_component += f"e^{'{' + str(x2) + 'x}'}"
                    if x3 != 0:
                        x3_component += f"e^{'{' + str(x3) + 'x}'}"
                    answers.append(
                        x1_component + "+" + x2_component + "+" + x3_component
                    )
        if return_df:
            return pd.DataFrame({"equation": equations, "answer": answers})
        return (equations, answers)

    def create_linear_combination(self, length: int = 2) -> tuple[list[str], list[str]]:
        tex_funcs = []
        sp_funcs = []
        for subset in combinations(self.func2tex.keys(), length):
            for perm in permutations(subset):
                sp, tex = None, None
                for func_sp in perm:
                    if sp is None:
                        sp = func_sp
                        tex = self.func2tex[func_sp]
                    else:
                        op = random.choice(["+", "-"])
                        if op == "+":
                            sp += func_sp
                            tex += " + " + self.func2tex[func_sp]
                        else:
                            sp -= func_sp
                            tex += " - " + self.func2tex[func_sp]
                tex_funcs.append(tex)
                sp_funcs.append(func_sp)
        return sp_funcs, tex_funcs

    @staticmethod
    # TODO: make work
    def simplyfy(text):
        return (
            text.replace("1y^{\prime\prime}", "y^{\prime\prime}")
            .replace("1y^{\prime}", "y^{\prime}")
            .replace("1y", "y")
            .replace("+ -", "-")
        )

    
    def linear_inhomogeneous_second_order(
        self,
        a_range: list  = range(-10, 11),
        b_range: list  = range(-10, 11),
        c_range: list  = range(-10, 11),
        return_df: bool = False,
        rhs_length: int = 2,
    ) -> pd.DataFrame | tuple[list[str], list[str]]:
        """
        Generates second order inhomogeneous linear equations
        ay^{\prime\prime} + by^{\prime} + cy = f(x)
        where f(x) is linear combination with 'length' terms that are base funcs,
        a, b and c from the corresponding ranges
        """
        equations = []
        answers = []
        for a in a_range:
            if a == 0:
                continue
            for b in b_range:
                for c in c_range:
                    for rhs, tex in zip(*self.create_linear_combination(rhs_length)):
                        y = Function("y")
                        lhs = (
                            a * y(self.x).diff(self.x, self.x)
                            + b * y(self.x).diff(self.x)
                            + c * y(self.x)
                        )
                        equation = Eq(lhs, rhs)
                        try:
                            ans = dsolve(equation)
                        except Exception as e:
                            print(e)
                        else:
                            equations.append(
                                Generator.simplyfy(
                                    str(a)
                                    + "y^{\prime\prime} + "
                                    + str(b)
                                    + "y^{\prime} + "
                                    + str(c)
                                    + "y = "
                                    + tex
                                )
                            )
                            print(equations[-1])
                            answers.append(latex(ans))
                        
        print(len(equations), len(answers))                
        if return_df:
            return pd.DataFrame({"equation": equations, "answer": answers})
        return (equations, answers)

    @classmethod
    def linear_inhomogeneous_third_order(
        self,
        a_range: list  = range(-10, 11),
        b_range: list  = range(-10, 11),
        c_range: list  = range(-10, 11),
        d_range: list  = range(-10, 11),
        return_df: bool = False,
        rhs_length: int = 2,
    ) -> pd.DataFrame | tuple[list[str], list[str]]:
        """
        Generates second order inhomogeneous linear equations
        ay^{\prime\prime\prime} + by^{\prime\prime} + cy^{\prime} + dy = f(x)
        where f(x) is linear combination with 'length' terms that are base funcs
        (a, b, c, d) from the corresponding ranges
        """
        equations = []
        answers = []
        for a in a_range:
            if a == 0:
                continue
            for b in b_range:
                for c in c_range:
                    for d in d_range:
                        for rhs, tex in self.create_linear_combination(rhs_length):
                            y = Function("y")
                            lhs = (
                                a * y(self.x).diff(self.x, self.x, self.x)
                                + b * y(self.x).diff(self.x, self.x)
                                + c * y(self.x).diff(self.x)
                                + d * y(self.x)
                            )
                            equation = Eq(lhs, rhs)
                            try: 
                                answer = dsolve(equation)
                            except Exception as e:
                                print(e)
                            else:
                                answers.append(latex(answer))
                                print(equations[-1], answers[-1])
                                equations.append(
                                        Generator.simplyfy(
                                            str(a)
                                            + "y^{\prime\prime\prime} + "
                                            + str(b)
                                            + "y^{\prime\prime} + "
                                            + str(c)
                                            + "y^{\prime} + "
                                            + str(d)
                                            + "y = "
                                            + tex
                                        )
                                    )
        if return_df:
            return pd.DataFrame({"equation": equations, "answer": answers})
        return (equations, answers)

    @staticmethod
    def equation(args: list[int | float]) -> str:
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

    @staticmethod
    def deriv(args: list[int | float]) -> str:
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
    
    @staticmethod
    def generate_polynomial(d: dict = None, start=-100, end=100, return_df=False) -> pd.DataFrame | tuple[list[str], list[str]]:
        '''
        generates equations of the form y^{\prime} = poly(x)
        param d should contain pairs (key, value)
        where key is degree of equation,
        value - amount of equations you want to generate,
        start and end - range of the coefficients
        '''
        if d is None:
            d = {i : random.randint(10, 100) for i in range(2, 10)}
        equations = []
        answers = []
        for key, val in d.items():
            for _ in range(val):
                args = random.sample(range(start, end), key)
                equations.append('y^{\prime}=' + re.sub(r'\+-', '-', Generator.deriv(args)))
                answers.append('y=' + re.sub(r'\+-', '-', Generator.equation(args)))
        if return_df:
            return pd.DataFrame({"equation": equations, "answer": answers})
        return (equations, answers)
