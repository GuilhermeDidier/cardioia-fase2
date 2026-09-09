"""CardioIA - Fase 2, Parte 1: extracao de sintomas e sugestao de diagnostico.

Le os relatos em `data/relatos_pacientes.txt`, procura nas frases as expressoes
do mapa de conhecimento (`data/mapa_conhecimento.csv`) e sugere um diagnostico.

O mapa tem tres colunas -- sintoma_1 | sintoma_2 | doenca_associada -- e cada
linha e uma *regra*: duas expressoes que, juntas, apontam para uma doenca.

Uso:
    python src/extracao.py
    python src/extracao.py --relato "sinto dor no peito quando subo escada"
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RELATOS = RAIZ / "data" / "relatos_pacientes.txt"
MAPA = RAIZ / "data" / "mapa_conhecimento.csv"

# Palavras que invalidam um sintoma encontrado logo depois delas.
# "nao sinto dor no peito" nao pode contar como dor no peito.
NEGACOES = ("nao", "sem", "nunca", "nego", "nenhum", "nenhuma")
JANELA_NEGACAO = 3  # em palavras


def normalizar(texto: str) -> str:
    """Minusculas, sem acento e sem pontuacao -- para casar texto com o mapa."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def carregar_relatos(caminho: Path = RELATOS) -> list[str]:
    """Le o .txt de relatos, ignorando comentarios (#) e linhas vazias."""
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in linhas if l.strip() and not l.startswith("#")]


def carregar_mapa(caminho: Path = MAPA) -> pd.DataFrame:
    """Le o mapa de conhecimento e ja guarda as expressoes normalizadas."""
    mapa = pd.read_csv(caminho)
    esperado = {"sintoma_1", "sintoma_2", "doenca_associada"}
    if not esperado.issubset(mapa.columns):
        raise ValueError(f"O mapa precisa das colunas {esperado}")
    mapa["expr_1"] = mapa["sintoma_1"].map(normalizar)
    mapa["expr_2"] = mapa["sintoma_2"].map(normalizar)
    return mapa


def _negado(texto_norm: str, inicio: int) -> bool:
    """True se houver negacao nas palavras imediatamente antes da expressao."""
    anteriores = texto_norm[:inicio].split()[-JANELA_NEGACAO:]
    return any(p in NEGACOES for p in anteriores)


def encontrar_expressoes(texto: str, expressoes: set[str]) -> set[str]:
    """Devolve as expressoes do mapa presentes no texto (ignorando negadas)."""
    texto_norm = normalizar(texto)
    achadas = set()
    for expr in expressoes:
        for m in re.finditer(re.escape(expr), texto_norm):
            if not _negado(texto_norm, m.start()):
                achadas.add(expr)
                break
    return achadas


@dataclass
class Hipotese:
    """Uma doenca candidata, com a pontuacao e a evidencia que a sustenta."""

    doenca: str
    pontuacao: float
    regras_completas: int
    sintomas: list[str] = field(default_factory=list)


def pontuar(texto: str, mapa: pd.DataFrame) -> list[Hipotese]:
    """Ranqueia as doencas do mapa para um relato.

    Pontuacao = nº de expressoes distintas encontradas para a doenca
                + 2 x nº de regras em que *as duas* expressoes apareceram.

    Contar expressoes distintas (e nao linhas do mapa) evita que uma queixa
    generica como "dor no peito", presente em varias linhas de Infarto, venca
    por repeticao. O bonus premia a evidencia combinada, que e o que o mapa
    de fato codifica.
    """
    expressoes = set(mapa["expr_1"]) | set(mapa["expr_2"])
    achadas = encontrar_expressoes(texto, expressoes)

    hipoteses: list[Hipotese] = []
    for doenca, regras in mapa.groupby("doenca_associada"):
        vistas: set[str] = set()
        completas = 0
        for _, r in regras.iterrows():
            tem_1, tem_2 = r["expr_1"] in achadas, r["expr_2"] in achadas
            if tem_1:
                vistas.add(r["expr_1"])
            if tem_2:
                vistas.add(r["expr_2"])
            if tem_1 and tem_2:
                completas += 1
        if vistas:
            hipoteses.append(
                Hipotese(doenca, len(vistas) + 2 * completas, completas, sorted(vistas))
            )
    return sorted(hipoteses, key=lambda h: (-h.pontuacao, h.doenca))


def sugerir_diagnostico(texto: str, mapa: pd.DataFrame) -> Hipotese | None:
    """A hipotese mais bem pontuada, ou None se nenhum sintoma foi reconhecido."""
    ranking = pontuar(texto, mapa)
    return ranking[0] if ranking else None


def relatorio(texto: str, mapa: pd.DataFrame, alternativas: int = 2) -> str:
    """Texto formatado com sintomas extraidos, hipotese principal e alternativas."""
    ranking = pontuar(texto, mapa)
    if not ranking:
        return "  Sintomas extraidos: (nenhum reconhecido)\n  -> Sem hipotese: relato fora do mapa de conhecimento."

    principal = ranking[0]
    linhas = [
        f"  Sintomas extraidos: {', '.join(principal.sintomas)}",
        f"  -> Hipotese: {principal.doenca} "
        f"(pontuacao {principal.pontuacao:g}, {principal.regras_completas} regra(s) completa(s))",
    ]
    outras = ranking[1 : 1 + alternativas]
    if outras:
        linhas.append(
            "     Alternativas: "
            + "; ".join(f"{h.doenca} ({h.pontuacao:g})" for h in outras)
        )
    if len(ranking) > 1 and ranking[1].pontuacao == principal.pontuacao:
        linhas.append("     ATENCAO: empate na pontuacao -- caso ambiguo, exige avaliacao humana.")
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrai sintomas e sugere diagnostico.")
    ap.add_argument("--relato", help="analisa uma frase avulsa em vez do arquivo")
    ap.add_argument("--relatos", type=Path, default=RELATOS)
    ap.add_argument("--mapa", type=Path, default=MAPA)
    args = ap.parse_args()

    mapa = carregar_mapa(args.mapa)
    textos = [args.relato] if args.relato else carregar_relatos(args.relatos)

    print(f"Mapa de conhecimento: {len(mapa)} regras, "
          f"{mapa.doenca_associada.nunique()} doencas.\n")
    for i, texto in enumerate(textos, 1):
        print(f"[Relato {i}] {texto}")
        print(relatorio(texto, mapa))
        print()


if __name__ == "__main__":
    main()
