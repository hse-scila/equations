import pandas as pd

def second_order_equation(x_1, x_2):
    if x_1 == x_2:
        raise Exception("Equal roots are not supported yet")
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

def generate_second_order(start=-10, end=11, return_df=False):
    equations = []
    answers = []
    for x1 in range(start, end):
        for x2 in range(start, end):
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
            # print(f"Answer for equation {equations[-1]} is {answers[-1]}")
    if return_df:
        return pd.DataFrame({"equation": equations, "answer": answers})
    return (equations, answers)

