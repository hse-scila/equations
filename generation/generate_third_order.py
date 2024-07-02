import pandas as pd

def third_order_equation(x_1, x_2, x_3):
    """ 
    C_1 * e^x1 + C_2 * e^x2 + C_3 * e^x3 
    is solution for 
    y^{\prime\prime\prime} + ay^{\prime\prime} + by^{\prime} + cy = 0
    """
    if x_1 == x_2 or x_1 == x_3 or x_2 == x_3:
        raise Exception("Roots should not be equal")
    a = -x_1 - x_2 - x_3
    b = x_1 * x_2 + x_2 * x_3 + x_1 * x_3
    c = x_1 * x_2 * x_3
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

def generate_third_order(start=-10, end=11):
    equations = []
    answers = []
    for x1 in range(start, end):
        for x2 in range(start, end):
            for x3 in range(start, end):
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
                answers.append(x1_component + "+" + x2_component + "+" + x3_component)
                # print(f"Answer for equation {equations[-1]} is {answers[-1]}")
    return pd.DataFrame({"equation": equations, "answer": answers})


