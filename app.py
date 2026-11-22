"""Demonstração: streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src.dados import RAIZ
from src.previsao import LIMITES, OPCOES, carregar_modelo, prever, validar

st.set_page_config(page_title="Prioridade de Contato", page_icon="📞", layout="wide")

PADRAO = {"age": 35, "job": "technician", "marital": "married", "education": "professional.course",
          "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "campaign": 1,
          "previous": 0, "poutcome": "nonexistent"}
EXEMPLOS = {
    "Técnico(a), 35 anos, sem histórico": PADRAO,
    "Aposentado(a), 68 anos, já contratou antes": {
        **PADRAO, "age": 68, "job": "retired", "education": "basic.4y", "previous": 1,
        "poutcome": "success"},
    "Operário(a), 42 anos, telefone fixo": {
        **PADRAO, "age": 42, "job": "blue-collar", "education": "basic.9y", "default": "unknown",
        "contact": "telephone", "campaign": 4},
}


@st.cache_resource
def obter_modelo():
    return carregar_modelo()


@st.cache_data
def obter_resultados():
    pasta = RAIZ / "resultados"
    return (json.loads((pasta / "metricas.json").read_text(encoding="utf-8")),
            pd.read_csv(pasta / "comparacao_modelos.csv"))


def formatar(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def aplicar_exemplo():
    for campo, valor in EXEMPLOS[st.session_state.exemplo].items():
        st.session_state[campo] = valor


def escolher(rotulo: str, campo: str):
    return st.selectbox(rotulo, list(OPCOES[campo]), format_func=OPCOES[campo].get, key=campo)


def numero(rotulo: str, campo: str, ajuda: str | None = None):
    minimo, maximo = LIMITES[campo]
    return int(st.number_input(rotulo, minimo, maximo, step=1, key=campo, help=ajuda))


for chave, valor in PADRAO.items():
    st.session_state.setdefault(chave, valor)

pacote = obter_modelo()
resumo, comparacao = obter_resultados()

st.title("📞 Quem devemos contatar primeiro?")
st.caption("Estima a chance de um cliente contratar um depósito a prazo, antes da ligação.")
aba_prever, aba_modelo = st.tabs(["Simular cliente", "Sobre o modelo"])

with aba_prever:
    st.selectbox("Carregar um exemplo", list(EXEMPLOS), key="exemplo", on_change=aplicar_exemplo)
    cliente = {}
    perfil, financeiro, campanha = st.columns(3)
    with perfil:
        st.subheader("Perfil")
        cliente["age"] = numero("Idade", "age")
        cliente["job"] = escolher("Profissão", "job")
        cliente["marital"] = escolher("Estado civil", "marital")
        cliente["education"] = escolher("Escolaridade", "education")
    with financeiro:
        st.subheader("Situação financeira")
        cliente["default"] = escolher("Crédito em atraso?", "default")
        cliente["housing"] = escolher("Financiamento imobiliário?", "housing")
        cliente["loan"] = escolher("Empréstimo pessoal?", "loan")
    with campanha:
        st.subheader("Contato")
        cliente["contact"] = escolher("Canal", "contact")
        cliente["campaign"] = numero("Ligações nesta campanha", "campaign",
                                     "Conta a ligação que será feita agora.")
        cliente["previous"] = numero("Contatos antes desta campanha", "previous")
        cliente["poutcome"] = escolher("Campanha anterior", "poutcome")

    st.divider()
    erros = validar(cliente)
    if erros:
        for erro in erros:
            st.error(erro)
    else:
        r = prever(cliente, pacote)
        a, b, c = st.columns(3)
        a.metric("Chance estimada de contratar", f"{formatar(100 * r['probabilidade'], 1)}%")
        b.metric("Prioridade", r["prioridade"])
        c.metric("Comparado à média", f"{formatar(r['vezes_a_media'], 1)} vezes")
        mostrar = {"Alta": st.success, "Média": st.warning, "Baixa": st.info}[r["prioridade"]]
        adesao = resumo["faixas"][r["prioridade"]]["adesao"]
        mostrar(f"**{r['acao']}** No teste, {round(100 * adesao)} de cada 100 pessoas "
                "nesta faixa contrataram.")
        for aviso in r["alertas"]:
            st.warning(f"Resultado menos confiável: {aviso}")
    st.caption("Estimativa baseada em campanhas de 2008 a 2010 de um banco português. "
               "Serve para ordenar contatos, não para negar produtos ou atendimento.")

with aba_modelo:
    t, tempo = resumo["teste"], resumo["tempo"]
    st.subheader(f"Modelo escolhido: {resumo['modelo']}")
    colunas = st.columns(4)
    colunas[0].metric("AUC-ROC", formatar(t["auc_roc"]))
    colunas[1].metric("AUC-PR", formatar(t["auc_pr"]),
                      help=f"Sem modelo seria {formatar(t['taxa_real'])}.")
    colunas[2].metric("Adesões encontradas", f"{formatar(100 * t['revocacao'], 1)}%",
                      help="Parcela de quem contratou que ficou na prioridade alta.")
    colunas[3].metric("Ganho sobre o acaso", f"{formatar(t['lift'], 2)} vezes")

    st.markdown("#### Comparação na validação cruzada")
    tabela = comparacao[["modelo", "auc_pr", "auc_roc", "brier"]].copy()
    for coluna in ["auc_pr", "auc_roc", "brier"]:
        tabela[coluna] = tabela[coluna].map(formatar)
    tabela.columns = ["Modelo", "AUC-PR", "AUC-ROC", "Brier (menor é melhor)"]
    st.dataframe(tabela, hide_index=True, width="stretch")

    figuras = st.columns(3)
    for coluna, nome in zip(figuras, ["matriz_confusao", "calibracao", "importancia"]):
        coluna.image(str(RAIZ / "resultados" / "figuras" / f"{nome}.png"), width="stretch")

    st.warning(
        f"**Limite importante:** ao treinar no passado e testar nos 20% mais recentes, a AUC-ROC "
        f"é {formatar(tempo['auc_roc'])}. Nesse período o modelo previu em média "
        f"{formatar(100 * tempo['probabilidade_media'], 1)}% de adesão, e a real foi "
        f"{formatar(100 * tempo['taxa_real'], 1)}%. Para uso atual, seria preciso retreinar com "
        "dados atuais.")
