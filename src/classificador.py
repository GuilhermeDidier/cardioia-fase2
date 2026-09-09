"""CardioIA - Fase 2, Parte 2: classificador de risco (triagem).

Vetoriza as frases com TF-IDF e treina um classificador que separa
"alto risco" de "baixo risco", como numa triagem que decide quem e atendido
primeiro. A avaliacao e deliberadamente desconfiada: alem da acuracia no teste,
roda validacao cruzada, compara com uma linha de base burra e submete o modelo
a um conjunto-desafio de frases dificeis.

Uso:
    python src/classificador.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .extracao import normalizar

RAIZ = Path(__file__).resolve().parent.parent
BASE = RAIZ / "data" / "frases_risco.csv"
DESAFIO = RAIZ / "data" / "frases_desafio.csv"

POSITIVO = "alto risco"  # classe cujo erro custa caro: e o falso negativo
SEMENTE = 42


def carregar_base(caminho: Path = BASE) -> pd.DataFrame:
    base = pd.read_csv(caminho)
    base["frase_norm"] = base["frase"].map(normalizar)
    return base


def construir_pipeline(modelo: str = "logistica") -> Pipeline:
    """TF-IDF (palavras + bigramas) seguido de um classificador simples."""
    classificadores = {
        "logistica": LogisticRegression(max_iter=1000, random_state=SEMENTE),
        "arvore": DecisionTreeClassifier(random_state=SEMENTE),
        "burro": DummyClassifier(strategy="most_frequent"),
    }
    if modelo not in classificadores:
        raise ValueError(f"modelo deve ser um de {list(classificadores)}")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", classificadores[modelo]),
        ]
    )


def avaliar(modelo: str = "logistica", base: pd.DataFrame | None = None) -> dict:
    """Treina e avalia: split estratificado + validacao cruzada 5-fold."""
    base = carregar_base() if base is None else base
    X, y = base["frase_norm"], base["situacao"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEMENTE
    )
    pipe = construir_pipeline(modelo).fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    cv = cross_val_score(
        construir_pipeline(modelo),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEMENTE),
        scoring="accuracy",
    )
    rotulos = sorted(y.unique())
    return {
        "modelo": modelo,
        "pipeline": pipe,
        "acuracia_teste": (y_pred == y_te).mean(),
        "cv_media": cv.mean(),
        "cv_desvio": cv.std(),
        "relatorio": classification_report(y_te, y_pred, zero_division=0),
        "matriz": pd.DataFrame(
            confusion_matrix(y_te, y_pred, labels=rotulos),
            index=[f"real: {r}" for r in rotulos],
            columns=[f"previsto: {r}" for r in rotulos],
        ),
        "y_teste": y_te,
        "y_previsto": y_pred,
    }


def termos_influentes(pipe: Pipeline, n: int = 10) -> pd.DataFrame:
    """Os termos que mais empurram a decisao para cada lado (so p/ logistica)."""
    vocab = pipe.named_steps["tfidf"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    classes = pipe.named_steps["clf"].classes_
    ordem = coefs.argsort()
    return pd.DataFrame(
        {
            f"puxa p/ '{classes[0]}'": vocab[ordem[:n]],
            f"puxa p/ '{classes[1]}'": vocab[ordem[-n:][::-1]],
        }
    )


def prever(pipe: Pipeline, frases: list[str]) -> pd.DataFrame:
    """Classifica frases novas, com a probabilidade de 'alto risco'."""
    norm = [normalizar(f) for f in frases]
    pred = pipe.predict(norm)
    idx = list(pipe.named_steps["clf"].classes_).index(POSITIVO)
    prob = pipe.predict_proba(norm)[:, idx]
    return pd.DataFrame({"frase": frases, "previsto": pred, f"p({POSITIVO})": prob.round(3)})


def testar_desafio(pipe: Pipeline, caminho: Path = DESAFIO) -> pd.DataFrame:
    """Roda o modelo no conjunto-desafio e marca os acertos."""
    desafio = pd.read_csv(caminho)
    saida = prever(pipe, desafio["frase"].tolist())
    saida["esperado"] = desafio["situacao"]
    saida["acertou"] = saida["previsto"] == saida["esperado"]
    saida["o_que_testa"] = desafio["o_que_testa"]
    return saida


def main() -> None:
    base = carregar_base()
    print(f"Base: {len(base)} frases -- {base.situacao.value_counts().to_dict()}\n")

    for modelo in ("burro", "arvore", "logistica"):
        r = avaliar(modelo, base)
        print(f"[{modelo:>9}] acuracia no teste: {r['acuracia_teste']:.3f} | "
              f"validacao cruzada: {r['cv_media']:.3f} +/- {r['cv_desvio']:.3f}")

    r = avaliar("logistica", base)
    print("\nRelatorio (regressao logistica, 21 frases de teste):")
    print(r["relatorio"])
    print(r["matriz"], "\n")

    pipe = construir_pipeline("logistica").fit(base["frase_norm"], base["situacao"])
    print("Termos mais influentes:")
    print(termos_influentes(pipe).to_string(index=False), "\n")

    desafio = testar_desafio(pipe)
    print(f"Conjunto-desafio: {desafio.acertou.sum()}/{len(desafio)} acertos")
    print(desafio[["frase", "esperado", "previsto", "p(alto risco)", "acertou"]].to_string(index=False))


if __name__ == "__main__":
    main()
