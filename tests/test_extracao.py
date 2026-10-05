"""Regressoes da Parte 1: negacao e regra de evidencia minima."""

import pytest

from src import extracao
from src.extracao import carregar_mapa, carregar_relatos, sugerir_diagnostico


@pytest.fixture(scope="module")
def mapa():
    return carregar_mapa()


def test_negacao_descarta_sintoma(mapa):
    assert sugerir_diagnostico("nao sinto dor no peito nem aperto no peito", mapa) is None


def test_nem_continua_negacao_anterior(mapa):
    # "nem" depois de "nao": a falta de ar tambem esta negada
    r = extracao.testar_desafio(mapa)
    rotina = r[r.relato.str.startswith("Não sinto dor no peito nem falta de ar")]
    assert rotina.sintomas.item() == ""


def test_nem_de_enfase_nao_nega(mapa):
    # "nem consigo" afirma a dificuldade; nao pode apagar o sintoma seguinte
    h = sugerir_diagnostico(
        "estou tão cansado que nem consigo subir um lance de escada, sinto aperto no peito", mapa)
    assert h is not None and h.doenca == "Angina"


def test_sintoma_isolado_nao_vira_diagnostico(mapa):
    assert sugerir_diagnostico("estou sem ar e com dor no peito", mapa) is None


def test_regra_completa_vence_expressoes_soltas(mapa):
    # AVC soma mais expressoes soltas (5 pts, 0 regras); Infarto tem 1 regra inteira
    h = sugerir_diagnostico(
        "dor no peito e aperto no peito e boca torta e braco dormente e confusao para falar "
        "e dificuldade para falar e dormencia", mapa)
    assert h is not None and h.doenca == "Infarto agudo do miocárdio"


def test_relatos_da_base_tem_hipotese(mapa):
    assert all(sugerir_diagnostico(r, mapa) for r in carregar_relatos())
