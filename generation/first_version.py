import pandas as pd

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

def linear_homogeneous_second_order(
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
            equations.append(second_order_equation(x1, x2))
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
                equations.append(third_order_equation(x1, x2, x3))
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