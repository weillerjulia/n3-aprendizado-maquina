"""Transforma um cliente novo em um resultado compreensível."""
import joblib
import pandas as pd

from src.dados import ENTRADAS, RAIZ

OPCOES = {
    "job": {
        "admin.": "Administrativo", "blue-collar": "Operário(a)", "technician": "Técnico(a)",
        "services": "Serviços", "management": "Gestão", "retired": "Aposentado(a)",
        "entrepreneur": "Empreendedor(a)", "self-employed": "Autônomo(a)",
        "housemaid": "Trabalho doméstico", "unemployed": "Desempregado(a)",
        "student": "Estudante", "unknown": "Não informado",
    },
    "marital": {"married": "Casado(a)", "single": "Solteiro(a)",
                "divorced": "Divorciado(a) ou viúvo(a)", "unknown": "Não informado"},
    "education": {
        "university.degree": "Ensino superior", "high.school": "Ensino médio",
        "professional.course": "Curso profissionalizante", "basic.9y": "Básico (9 anos)",
        "basic.6y": "Básico (6 anos)", "basic.4y": "Básico (4 anos)",
        "illiterate": "Não alfabetizado(a)", "unknown": "Não informado",
    },
    "default": {"no": "Não", "yes": "Sim", "unknown": "Não informado"},
    "housing": {"no": "Não", "yes": "Sim", "unknown": "Não informado"},
    "loan": {"no": "Não", "yes": "Sim", "unknown": "Não informado"},
    "contact": {"cellular": "Celular", "telephone": "Telefone fixo"},
    "poutcome": {"nonexistent": "Não participou", "failure": "Participou e não contratou",
                 "success": "Participou e contratou"},
}
# Limites observados na base: fora deles o modelo não tem como opinar
LIMITES = {"age": (17, 98), "campaign": (1, 56), "previous": (0, 7)}


def validar(cliente: dict) -> list[str]:
    faltando = [c for c in ENTRADAS if c not in cliente]
    if faltando:
        return [f"Campos ausentes: {', '.join(faltando)}."]
    erros = []
    for campo, (minimo, maximo) in LIMITES.items():
        valor = cliente[campo]
        if type(valor) is not int or not minimo <= valor <= maximo:
            erros.append(f"{campo} deve ser um inteiro de {minimo} a {maximo}.")
    for campo, opcoes in OPCOES.items():
        if cliente[campo] not in opcoes:
            erros.append(f"Valor inválido em {campo}: {cliente[campo]}.")
    if not erros and (cliente["previous"] == 0) != (cliente["poutcome"] == "nonexistent"):
        erros.append("Contatos anteriores e resultado da campanha anterior se contradizem: "
                     "use 0 contatos somente para quem não participou.")
    return erros


def alertas(cliente: dict) -> list[str]:
    """Entradas válidas, mas com poucos exemplos na base."""
    avisos = []
    if cliente["default"] == "yes":
        avisos.append("Só 3 pessoas da base têm crédito em atraso.")
    if cliente["education"] == "illiterate":
        avisos.append("Só 18 registros da base são de pessoas não alfabetizadas.")
    if cliente["age"] > 80 or cliente["campaign"] > 15 or cliente["previous"] > 3:
        avisos.append("Há poucos registros com valores numéricos tão altos.")
    return avisos


def carregar_modelo() -> dict:
    return joblib.load(RAIZ / "modelos" / "modelo.joblib")


def prever(cliente: dict, pacote: dict) -> dict:
    erros = validar(cliente)
    if erros:
        raise ValueError(" ".join(erros))
    entrada = pd.DataFrame([cliente])[ENTRADAS]
    prob = float(pacote["pipeline"].predict_proba(entrada)[0, 1])
    if prob >= pacote["cortes"]["alta"]:
        faixa, acao = "Alta", "Ligar primeiro."
    elif prob >= pacote["cortes"]["media"]:
        faixa, acao = "Média", "Ligar depois da lista de prioridade alta."
    else:
        faixa, acao = "Baixa", "Deixar para o fim da lista."
    return {"probabilidade": prob, "prioridade": faixa, "acao": acao,
            "vezes_a_media": prob / pacote["taxa_treino"], "alertas": alertas(cliente)}
