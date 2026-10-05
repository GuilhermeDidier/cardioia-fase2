"""Regressoes da Parte 2: o limiar aplicado deve concordar com a curva."""

from src.classificador import aplicar_limiar, carregar_base, construir_pipeline, curva_limiar


def test_aplicar_limiar_usa_probabilidade_crua():
    base = carregar_base()
    pipe = construir_pipeline().fit(base["frase_norm"], base["situacao"])
    frases = base["frase"].tolist()
    p = pipe.predict_proba(base["frase_norm"])[:, list(pipe.classes_).index("alto risco")]
    previsto = aplicar_limiar(pipe, frases, 0.45)
    assert previsto == ["alto risco" if x >= 0.45 else "baixo risco" for x in p]


def test_curva_limiar_baixar_corte_nao_perde_graves():
    curva = curva_limiar()
    assert curva["graves perdidos"].is_monotonic_decreasing
    assert curva["falsos positivos"].is_monotonic_increasing
