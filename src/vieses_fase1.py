"""CardioIA - Fase 2: vieses do dataset da Fase 1 que afetam a triagem por texto.

O dataset da Fase 1 (`data/pacientes_cardiacos_fase1.csv`) e o subconjunto
Cleveland do UCI Heart Disease: 303 pacientes encaminhados para cateterismo,
com o diagnostico confirmado por angiografia. Ele nao tem texto, mas tem a
variavel `tipo_dor_peito` -- o mesmo sinal em que o classificador da Parte 2
mais se apoia. Cruzar as duas coisas mostra quem a triagem por texto deixa
de ver.

Uso:
    python -m src.vieses_fase1
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PACIENTES = RAIZ / "data" / "pacientes_cardiacos_fase1.csv"

# Codificacao original do UCI (ver scripts/mapeamento_colunas.md no repo da Fase 1).
TIPOS_DOR = {1: "angina típica", 2: "angina atípica", 3: "dor não anginosa", 4: "assintomático"}
ASSINTOMATICO = 4  # sem dor toracica nenhuma
SEXOS = {0: "feminino", 1: "masculino"}


def carregar_pacientes(caminho: Path = PACIENTES) -> pd.DataFrame:
    """Le o CSV da Fase 1; `doente` = diagnostico > 0 (qualquer grau de estenose)."""
    df = pd.read_csv(caminho)
    df["doente"] = df["diagnostico"] > 0
    df["dor_no_peito"] = pd.Categorical(
        df["tipo_dor_peito"].map(TIPOS_DOR), categories=list(TIPOS_DOR.values()), ordered=True)
    df["sexo_desc"] = df["sexo"].map(SEXOS)
    if df["sexo_desc"].isna().any() or df["dor_no_peito"].isna().any():
        raise ValueError("codigo de sexo ou de tipo de dor fora da codificacao do UCI")
    return df


def prevalencia_por(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Pacientes, doentes e % de doentes em cada grupo da coluna."""
    t = df.groupby(coluna, observed=True)["doente"].agg(pacientes="count", doentes="sum")
    t["% doentes"] = (100 * t["doentes"] / t["pacientes"]).round(1)
    return t


def doentes_assintomaticos(df: pd.DataFrame) -> pd.DataFrame:
    """Entre os doentes, a fracao classificada como assintomatica (tipo 4).

    No UCI, "assintomatico" e o paciente sem dor toracica: nao relatou angina
    tipica, angina atipica nem dor nao anginosa.
    """
    doentes = df[df["doente"]]
    sem_dor = doentes["tipo_dor_peito"] == ASSINTOMATICO
    t = doentes.assign(assintomatico=sem_dor).groupby("sexo_desc")["assintomatico"].agg(
        doentes="count", sem_dor_no_peito="sum")
    t.loc["total"] = t.sum()
    t["% sem dor no peito"] = (100 * t["sem_dor_no_peito"] / t["doentes"]).round(1)
    return t


def main() -> None:
    df = carregar_pacientes()
    print(f"Dataset da Fase 1: {len(df)} pacientes, {df.doente.mean():.1%} com doença\n")
    print(prevalencia_por(df, "dor_no_peito").to_string(), "\n")
    print(prevalencia_por(df, "sexo_desc").to_string(), "\n")
    print(doentes_assintomaticos(df).to_string())


if __name__ == "__main__":
    main()
