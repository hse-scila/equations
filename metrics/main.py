import pandas as pd
import sympy as sp
import numpy as np
from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

def latex_to_func(latex_str, variable='x'):
    """
    Преобразует строку с математическим выражением в sympy выражение
    """
    try:
        # Заменяем некоторые LaTeX-специфичные конструкции на SymPy-совместимые
        expr_str = latex_str.replace('\\', '').replace('{', '').replace('}', '')
        expr = parse_expr(expr_str)
    except Exception as e:
        raise ValueError(f"Не удалось распарсить выражение: {latex_str}, ошибка: {e}")
    
    x = sp.symbols(variable)
    if not expr.has(x):
        raise ValueError(f"Функция не содержит переменной '{variable}': {latex_str}")
    
    return expr

def latex_equiv(latex_str1, latex_str2):
    """
    Возвращает 1 если функции, представленные строками идентичны и 0 иначе
    """
    try:
        expr1 = latex_to_func(latex_str1)
    except Exception as e:
        print(f"Не удалось распарсить выражение: {latex_str1}, ошибка: {e}")
        return 0
    try:
        expr2 = latex_to_func(latex_str2)
    except Exception as e:
        print(f"Не удалось распарсить выражение: {latex_str2}, ошибка: {e}")
        return 0

    return expr1 == expr2
        

def compute_norm(func1, func2, variable='x', a=-10, b=10, n_points=1000):
    """
    Считает L2 норму между двумя sympy функциями
    """
    x = sp.symbols(variable)
    
    f1 = lambdify(x, func1, 'numpy')
    f2 = lambdify(x, func2, 'numpy')
    
    x_vals = np.linspace(a, b, n_points)
    
    try:
        y1 = f1(x_vals)
        y2 = f2(x_vals)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Ошибка при вычислении функции: {e}")
    
    diff = y1 - y2
    return np.trapezoid(diff**2, x_vals) ** 0.5, np.max(np.abs(diff))
    
def compute_bleu(text1, text2):
    """
    Считает BLEU между двумя документами
    """
    tokens1 = list(text1)
    tokens2 = list(text2)
    
    smooth = SmoothingFunction().method1
    score = sentence_bleu(
        [tokens1],
        tokens2,
        smoothing_function=smooth,
        weights=(0.25, 0.25, 0.25, 0.25) 
    )
    return score

def compute_norm_for_dataframe(df):
    l2_norms = []
    inf_norms = []
    bleus = []
    is_eq = []
    for idx, row in df.iterrows():
        try:
            func1 = latex_to_func(row['predictions'])
            func2 = latex_to_func(row['answer'])
            l2_norm, inf_norm = compute_norm(func1, func2)
            l2_norms.append(l2_norm)
            inf_norms.append(inf_norm)
            is_eq.append(func1 == func2)
        except Exception as e:
            print(f"Ошибка в строке {idx}: {e}")
            l2_norms.append(np.nan)
            inf_norms.append(np.nan)
            is_eq.append(0)
        
        bleu = np.nan
        try:
            bleu = compute_bleu(row['answer'], row['predictions'])
        except Exception as e:
            print(f"Ошибка в строке {idx} при вычислении BLEU: {e}")
        bleus.append(bleu)
    
    df['l2_norm'] = l2_norms
    df['inf_norm'] = inf_norms
    df['bleus'] = bleus
    df['is_eq'] = is_eq
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь_к_файлу>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    df = pd.read_csv(file_path, escapechar="\\", on_bad_lines='warn')
    df = compute_norm_for_dataframe(df)
    
    output_path = file_path.replace('.', '_metrics.')
    df.to_csv(output_path, index=False)