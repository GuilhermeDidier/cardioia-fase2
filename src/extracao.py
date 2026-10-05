"""CardioIA - Fase 2, Parte 1: extracao de sintomas e sugestao de diagnostico.

Le os relatos em `data/relatos_pacientes.txt`, procura nas frases as expressoes
do mapa de conhecimento (`data/mapa_conhecimento.csv`) e sugere um diagnostico.

O mapa tem tres colunas -- sintoma_1 | sintoma_2 | doenca_associada -- e cada
linha e uma *regra*: duas expressoes que, juntas, apontam para uma doenca.

Uso:
    python src/extracao.py
    python src/extracao.py --relato "sinto dor no peito quando subo escada"
    python src/extracao.py --desafio
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
DESAFIO = RAIZ / "data" / "relatos_desafio.csv"
SEM_HIPOTESE = "(sem hipótese)"

# Palavras que invalidam um sintoma encontrado logo depois delas.
# "nao sinto dor no peito" nao pode contar como dor no peito.
NEGACOES = ("nao", "sem", "nunca", "nego", "nenhum", "nenhuma")
JANELA_NEGACAO = 3  # em palavras

# "nem" so nega quando continua uma negacao anterior: "nao sinto dor no peito
# NEM falta de ar". Sozinho ele costuma ser enfase e AFIRMA o sintoma:
# "estou tao cansado que nem consigo subir um lance de escada".
NEM = "nem"
JANELA_NEM = 8  # palavras antes do "nem" onde procurar a negacao que ele continua

# Uma regra do mapa e um *par* de expressoes. Sem nenhum par completo, o que
# sobra sao expressoes soltas ("dor no peito") que aparecem em varias doencas,
# e a escolha entre elas seria decidida pelo desempate -- ou seja, por nada.
MIN_REGRAS_COMPLETAS = 1


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
    palavras = texto_norm[:inicio].split()
    janela = palavras[-JANELA_NEGACAO:]
    if any(p in NEGACOES for p in janela):
        return True
    if NEM in janela:
        pos_nem = len(palavras) - len(janela) + janela.index(NEM)
        antes_do_nem = palavras[max(0, pos_nem - JANELA_NEM):pos_nem]
        return any(p in NEGACOES for p in antes_do_nem)
    return False


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


def _decidir(ranking: list[Hipotese]) -> Hipotese | None:
    """A mais bem pontuada entre as doencas com evidencia suficiente.

    So concorre quem tem ao menos MIN_REGRAS_COMPLETAS regra(s) inteira(s):
    uma doenca com varias expressoes soltas nao passa na frente de outra que
    tem um par completo do mapa.
    """
    elegiveis = [h for h in ranking if h.regras_completas >= MIN_REGRAS_COMPLETAS]
    return elegiveis[0] if elegiveis else None


def _sintomas(ranking: list[Hipotese]) -> list[str]:
    """Todas as expressoes reconhecidas no relato, de qualquer doenca."""
    return sorted(set().union(*(h.sintomas for h in ranking)))


def sugerir_diagnostico(texto: str, mapa: pd.DataFrame) -> Hipotese | None:
    """A hipotese sugerida, ou None se a evidencia nao basta (ver _decidir)."""
    return _decidir(pontuar(texto, mapa))


def relatorio(texto: str, mapa: pd.DataFrame, alternativas: int = 2) -> str:
    """Texto formatado com sintomas extraidos, hipotese principal e alternativas."""
    ranking = pontuar(texto, mapa)
    if not ranking:
        return "  Sintomas extraídos: (nenhum reconhecido)\n  -> Sem hipótese: relato fora do mapa de conhecimento."

    principal = _decidir(ranking)
    todos = _sintomas(ranking)
    if principal is None:
        return (
            f"  Sintomas extraídos: {', '.join(todos)}\n"
            "  -> Sem hipótese: evidência insuficiente (nenhuma regra do mapa disparou inteira).\n"
            "     Candidatas: " + "; ".join(f"{h.doenca} ({h.pontuacao:g})" for h in ranking[:3])
        )
    linhas = [
        f"  Sintomas extraídos: {', '.join(todos)}",
        f"  -> Hipótese: {principal.doenca} "
        f"(pontuação {principal.pontuacao:g}, {principal.regras_completas} regra(s) completa(s))",
    ]
    outras = [h for h in ranking if h is not principal][:alternativas]
    if outras:
        linhas.append(
            "     Alternativas: "
            + "; ".join(f"{h.doenca} ({h.pontuacao:g})" for h in outras)
        )
    if any(h.pontuacao == principal.pontuacao for h in outras):
        linhas.append("     ATENÇÃO: empate na pontuação -- caso ambíguo, exige avaliação humana.")
    return "\n".join(linhas)


def testar_desafio(mapa: pd.DataFrame, caminho: Path = DESAFIO) -> pd.DataFrame:
    """Roda o extrator nos relatos-desafio (escritos com o mapa congelado)."""
    desafio = pd.read_csv(caminho)
    linhas = []
    for _, d in desafio.iterrows():
        ranking = pontuar(d["relato"], mapa)
        h = _decidir(ranking)
        sugerido = h.doenca if h else SEM_HIPOTESE
        linhas.append({
            "relato": d["relato"],
            "esperado": d["esperado"],
            "sugerido": sugerido,
            "sintomas": ", ".join(_sintomas(ranking)),
            # compara normalizado: acento ou forma Unicode diferente nao e erro
            "acertou": normalizar(sugerido) == normalizar(d["esperado"]),
            "o_que_testa": d["o_que_testa"],
        })
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrai sintomas e sugere diagnostico.")
    ap.add_argument("--relato", help="analisa uma frase avulsa em vez do arquivo")
    ap.add_argument("--relatos", type=Path, default=RELATOS)
    ap.add_argument("--mapa", type=Path, default=MAPA)
    ap.add_argument("--desafio", action="store_true",
                    help="roda os relatos-desafio, escritos sem alterar o mapa")
    args = ap.parse_args()

    mapa = carregar_mapa(args.mapa)
    if args.desafio:
        r = testar_desafio(mapa)
        print(r[["esperado", "sugerido", "acertou", "sintomas"]].to_string(index=False))
        print(f"\nAcertos no desafio: {r.acertou.sum()}/{len(r)}")
        return
    textos = [args.relato] if args.relato else carregar_relatos(args.relatos)

    print(f"Mapa de conhecimento: {len(mapa)} regras, "
          f"{mapa.doenca_associada.nunique()} doenças.\n")
    for i, texto in enumerate(textos, 1):
        print(f"[Relato {i}] {texto}")
        print(relatorio(texto, mapa))
        print()


if __name__ == "__main__":
    main()
