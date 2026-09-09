# CardioIA — Fase 2: Diagnóstico Automatizado

Módulo de apoio ao diagnóstico do projeto **CardioIA** (FIAP). Duas peças que resolvem
perguntas diferentes a partir do texto que o paciente escreve:

| Parte | Pergunta | Abordagem |
|---|---|---|
| **1 — Extração de sintomas** | *Qual doença esses sintomas sugerem?* | Mapa de conhecimento + regras, com evidência explícita |
| **2 — Classificador de risco** | *Com que urgência atender?* | TF-IDF + Regressão Logística (Scikit-learn) |

O vocabulário clínico e os quadros usados aqui seguem o
[UCI Heart Disease Data Set](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Detrano et al., 1989), a base pública de referência para triagem de doença coronariana.

📹 **Vídeo de demonstração (4 min):** `<INSERIR LINK DO YOUTUBE — não listado>`

---

## Resultados medidos

**Parte 1 — extração e diagnóstico**

| Métrica | Valor |
|---|---|
| Relatos processados | 10 |
| Regras no mapa de conhecimento | 79, cobrindo 8 doenças |
| Acerto contra o gabarito interno | **10/10** |
| Menor margem para a 2ª hipótese | 4 pontos |

**Parte 2 — classificação de risco**

| Avaliação | Acurácia | O que ela mede |
|---|---|---|
| Linha de base (`DummyClassifier`) | 0,476 | O piso: chutar sempre a classe mais comum |
| Árvore de decisão | 0,571 | |
| **Regressão logística — teste sorteado** | **0,762** | Consistência interna da base |
| Regressão logística — validação cruzada 5-fold | 0,829 ± 0,107 | Estabilidade (o desvio é grande: base pequena) |
| **Regressão logística — conjunto-desafio** | **0,375** | Frases novas, escritas para atacar fraquezas conhecidas |

> **A distância entre 0,762 e 0,375 é o principal resultado deste trabalho.** A acurácia
> do teste media o quanto a base é homogênea, não o quanto o modelo entende de risco
> clínico. Toda a análise está na seção 6 do notebook da Parte 2.

![Matriz de confusão](docs/matriz_confusao.png)

Três dos onze casos de alto risco foram classificados como baixo risco. Numa triagem, é
o erro que custa caro — e o motivo de a métrica que governa este problema ser o **recall
de "alto risco"**, não a acurácia.

---

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/extracao.py           # Parte 1: processa os 10 relatos
python -m src.classificador      # Parte 2: treina, avalia e roda o conjunto-desafio

jupyter notebook notebooks/      # os dois notebooks, com toda a análise
```

Uma frase avulsa, sem editar arquivo:

```bash
python src/extracao.py --relato "sinto um aperto no peito quando faco esforco fisico"
# -> Hipotese: Angina (pontuacao 4, 1 regra(s) completa(s))
```

---

## Estrutura

```
cardioia-fase2/
├── data/
│   ├── relatos_pacientes.txt        # Parte 1 — 10 relatos de sintomas
│   ├── mapa_conhecimento.csv        # Parte 1 — 79 regras (sintoma_1 | sintoma_2 | doenca_associada)
│   ├── frases_risco.csv             # Parte 2 — 70 frases rotuladas (frase, situacao)
│   ├── frases_desafio.csv           # Parte 2 — 8 frases de estresse, fora do treino
│   └── frases_ampliacao.csv         # Parte 2 — 8 frases do experimento de correção da base
├── src/
│   ├── extracao.py                  # normalização, negação, extração e ranqueamento
│   └── classificador.py             # pipeline TF-IDF + modelo, avaliação, previsão
├── notebooks/
│   ├── parte1_extracao_sintomas.ipynb
│   └── parte2_classificador_risco.ipynb
├── docs/matriz_confusao.png
└── requirements.txt
```

Os notebooks **importam** `src/` em vez de repetir o código: a mesma lógica roda na
análise e na linha de comando, sem duas versões para manter em sincronia.

---

## Decisões técnicas

**O ranqueamento não soma linhas do mapa.** A pontuação de uma doença é
`nº de expressões distintas encontradas + 2 × nº de regras que dispararam inteiras`.
Somar linha a linha faria "dor no peito" — presente em 7 regras de infarto — vencer por
repetição, sem nenhuma evidência específica. Contar expressões distintas elimina isso; o
bônus premia a evidência combinada, que é o que o mapa de fato codifica.

**A extração trata negação; o classificador não.** `"não sinto dor no peito"` é
descartada corretamente pela Parte 1 e classificada como **alto risco com 75% de
confiança** pela Parte 2 — para o TF-IDF, "não" é só mais um token. As duas abordagens
falham em lugares diferentes, e nenhuma domina a outra.

**Divergimos de um exemplo do enunciado.** Ele sugere `"falta de ar" → Angina`.
Clinicamente, dispneia isolada é sinal de **insuficiência cardíaca**; angina se define
pela dor torácica desencadeada por esforço. Seguimos a classificação clínica e
registramos a divergência.

**Frases de baixo risco também falam em peito.** Se só as frases graves mencionassem
"peito" ou "dor", o classificador acertaria tudo procurando uma palavra, e a avaliação
não mediria nada. A sobreposição de vocabulário entre as classes é intencional.

**Corrigimos a base, não o modelo — e a acurácia caiu.** Adicionar 8 frases de negação e
apresentação atípica levou a acurácia de 0,762 para 0,625 e o desempenho no
conjunto-desafio de 3/8 para 4/8. Casos difíceis derrubam a métrica interna e melhoram o
sistema; quem otimizasse pela acurácia teria descartado a melhoria certa.

---

## Vieses conhecidos

| Viés | Origem | Efeito medido |
|---|---|---|
| Apresentação atípica sub-representada | As frases seguem o padrão clássico "dor no peito com irradiação" | Infarto atípico (mais frequente em mulheres) classificado como **baixo risco** |
| Vocabulário como proxy de gravidade | "leve", "pouco" só aparecem em frases de baixo risco | Quem minimiza a própria queixa é despriorizado |
| Base autoral | As 70 frases foram escritas e rotuladas pela mesma equipe | O modelo aprende o nosso jeito de escrever, não o do paciente |
| Ausência de negação no treino | Nenhuma frase original diz "não sinto" | Consulta de rotina vira urgência |

O paralelo com dados clínicos reais é direto: no UCI Heart Disease, a prevalência de
doença varia de **36% a 93%** entre as quatro coortes hospitalares que compõem a base —
um modelo treinado em uma delas erra sistematicamente nas outras. **Quem escreve os dados
decide o que o modelo enxerga**, e isso não se corrige trocando de algoritmo.

---

## Limites de uso

Exercício acadêmico sobre **frases fictícias escritas por nós**. Nenhum relato é de
paciente real. O sistema não foi validado clinicamente, erra casos graves de forma
sistemática e documentada, e **não é dispositivo médico**: não deve orientar decisão
sobre pessoa nenhuma.

---

## Equipe

| Nome | RM |
|---|---|
| `<PREENCHER>` | `<PREENCHER>` |

FIAP — Inteligência Artificial · Projeto CardioIA · Fase 2
