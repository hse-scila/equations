import pandas as pd
import sympy as sp
import numpy as np
from latex2sympy2 import latex2sympy
from sympy import lambdify
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')

def latex_to_func(latex_str, variable='x'):
    try:
        expr = latex2sympy(latex_str)
    except Exception as e:
        raise ValueError(f"Не удалось распарсить LaTeX: {latex_str}, ", e)
    
    x = sp.symbols(variable)
    if not expr.has(x):
        raise ValueError(f"Функция не содержит переменной '{variable}': {latex_str}")
    
    return expr

def compute_norm(func1, func2, variable='x', a=-10, b=10, n_points=1000):
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
    for idx, row in df.iterrows():
        try:
            func1 = latex_to_func(row['prediction'])
            func2 = latex_to_func(row['label'])
            l2_norm, inf_norm = compute_norm(func1, func2)
            l2_norms.append(l2_norm)
            inf_norms.append(inf_norm)
        except Exception as e:
            print(f"Ошибка в строке {idx}: {e}")
            l2_norms.append(np.nan)
            inf_norms.append(np.nan)
        
        bleu = np.nan
        try:
            bleu = compute_bleu(row['label'], row['prediction'])
        except Exception as e:
            print(f"Ошибка в строке {idx} при вычислении BLEU: {e}")
        bleus.append(bleu)
    
    df['l2_norm'] = l2_norms
    df['inf_norm'] = inf_norms
    df['bleus'] = bleus
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь_к_файлу>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    df = pd.read_csv(file_path)
    df = compute_norm_for_dataframe(df)
    
    output_path = file_path.replace('.', '_metrics.')
    df.to_csv(output_path, index=False)
    