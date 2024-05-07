from torchmetrics.text import BLEUScore
from Training import predict


def bleu_score(preds, refs):
    bleu = BLEUScore()
    return bleu(preds, [[i] for i in refs]).item()

def accuracy(preds, refs):
    equal_count = 0
    for i in range(len(refs)):
        if preds[i] == refs[i]:
            equal_count += 1
    return equal_count / len(refs)

valid_eqs = [
    "2xy\mathrm{d}x + (x^2 - y^2)\mathrm{d}y = 0",  # в полных дифференциалах
    "\frac{3x^2 + y^2}{y^2}\mathrm{d}x - \frac{2x^3 + 5y}{y^3}\mathrm{d}y",  # в полных дифференциалах
    "y^{\prime}=\mathrm{tg}{\frac{y}{x}}+{\frac{y}{x}}",  # однородное
    "y^{\prime}=\cos^{2}{\frac{y}{x}}+{\frac{y}{x}}",  # однородное
    "y^{\prime}-y={\frac{e^{x}}{x^{2}}}",  # линейное 1-го порядка
    "(2x+y^{2})y^{\prime}=y",  # линейное 1-го порядка
    "y y^{\prime3}+x=1",  # не разрешенное относительно производной
    "y^{\prime^{3}}+y^{2}=y y^{\prime}(y^{\prime}+1)",  # не разрешенное относительно производной
    "2y^{\prime}-\frac{y}{x}=\frac{4x^{2}}{y}",  # уравнение Бернулли
    "xy^{\prime}-2y={\frac{x}{y}}",  # уравнение Бернулли
    "2y^{\prime\prime}+3y^{\prime}-5y=10",  # неоднородные линейные
    "y^{\prime\prime}-2y^{\prime}-8y=x^{2}+3",  # неоднородные линейные
    "{\frac{x\,d x+y\,d y}{y\,\overline{{{1+x^{2}+y^{2}}}}}}+{\frac{y\,d x-x\,d y}{x^{2}+y^{2}}}=0",  # интегрирующий множитель
    "(x^{2}y^{2}-1)\,d y+2x y^{3}\,d x=0",  # интегрирующий множитель
]
valid_eqs_answers = [
    "3x^2 - y^3 = C",
    "x + \frac{x^3}{y^2} + \frac{5}{y} = C",
    "y=x\arcsin(C x)",
    "y(x)=x\tan^{-1}(c_{1}+\log(x))",
    "y(x)=c_{1}\,e^{x}-{\frac{e^{x}}{x}}",
    "x=y^{2}(\ln y + C)",
    "(x-1)^{4/3}+y^{4/3}=C",
    "4y=(x+C)^{2}",
    "y(x)=-{\sqrt{x}}\;{\sqrt{c_{1}+2x^{2}}}",
    "y(x)={\frac{\sqrt{x}\;{\sqrt{c_{1}\,x^{3}-2}}}{\sqrt{3}}}",
    "y(x)=c_{1}\;e^{-(5x)/2}+c_{2}\;e^{x}-2",
    "y(x)=c_{1}\;e^{-2x}+c_{2}\;e^{4x}-{\frac{x^{2}}{8}}+{\frac{x}{16}}-{\frac{27}{64}}",
    "{\sqrt{{1+x^{2}+y^{2}}}}+\arctan{\frac{x}{y}}=C",
    "x^{2}y+{\frac{1}{y}}=C",
]

def predict_valid_eqs(model, file_name, tokenizer, tokenizer_type, device='cuda', only_decoder=False):
    preds = []
    for eq in valid_eqs:
        preds.append(predict(eq, model, tokenizer, tokenizer_type, device=device, only_decoder=only_decoder))
    d = dict(zip(valid_eqs_answers, preds))
    return d