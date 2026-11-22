"""Carga, limpeza e separação dos dados."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / "dados" / "bank-additional-full.csv"
SEMENTE = 42

PERFIL = ["age", "job", "marital", "education", "default", "housing", "loan"]
NUMERICAS = ["age", "campaign", "previous"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
ENTRADAS = NUMERICAS + CATEGORICAS

# Colunas que ficam fora do modelo e o motivo de cada uma
EXCLUIDAS = {
    "duration": "Só é conhecida depois da ligação (vazamento).",
    "pdays": "999 é um código; o histórico já é representado por poutcome e previous.",
    "month": "Identifica a época da campanha, não o cliente.",
    "day_of_week": "Depende de quando se liga, não de quem é o cliente.",
    "emp.var.rate": "Indicador econômico: igual para os clientes da mesma época.",
    "cons.price.idx": "Indicador econômico: igual para os clientes da mesma época.",
    "cons.conf.idx": "Indicador econômico: igual para os clientes da mesma época.",
    "euribor3m": "Indicador econômico: igual para os clientes da mesma época.",
    "nr.employed": "Indicador econômico: igual para os clientes da mesma época.",
}


def carregar() -> pd.DataFrame:
    """Lê a base, remove duplicatas exatas e cria o alvo. A ordem cronológica é mantida."""
    dados = pd.read_csv(ARQUIVO, sep=";").drop_duplicates().reset_index(drop=True)
    dados["alvo"] = (dados["y"] == "yes").astype(int)
    return dados


def grupos_de_perfil(dados: pd.DataFrame) -> pd.Series:
    """A base não tem ID de cliente, então perfis idênticos são tratados como um grupo."""
    return dados.groupby(PERFIL, sort=False).ngroup()


def dobras(dados: pd.DataFrame) -> list:
    """5 dobras estratificadas, sem repetir perfil entre treino e avaliação."""
    divisor = StratifiedGroupKFold(5, shuffle=True, random_state=SEMENTE)
    return list(divisor.split(dados, dados["alvo"], grupos_de_perfil(dados)))


def separar(dados: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80% treino e 20% teste."""
    treino, teste = dobras(dados)[0]
    return dados.iloc[treino].reset_index(drop=True), dados.iloc[teste].reset_index(drop=True)
