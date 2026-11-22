"""Treina, escolhe e avalia o modelo: python -m src.treinar"""
import json

import joblib
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_predict

from src.avaliacao import erros_por_segmento, metricas
from src.dados import ENTRADAS, RAIZ, SEMENTE, carregar, dobras, separar
from src.modelos import candidatos, criar_pipeline

SAIDA = RAIZ / "resultados"


def comparar(treino: pd.DataFrame):
    """Validação cruzada no treino. Vence a maior AUC-PR média."""
    cv = dobras(treino)
    linhas, buscas = [], {}
    for nome, (algoritmo, grade) in candidatos().items():
        busca = GridSearchCV(
            criar_pipeline(algoritmo), grade, cv=cv, refit="auc_pr",
            scoring={"auc_pr": "average_precision", "auc_roc": "roc_auc", "brier": "neg_brier_score"},
        ).fit(treino[ENTRADAS], treino["alvo"])
        r, i = busca.cv_results_, busca.best_index_
        linhas.append({
            "modelo": nome,
            "auc_pr": r["mean_test_auc_pr"][i], "desvio": r["std_test_auc_pr"][i],
            "auc_roc": r["mean_test_auc_roc"][i], "brier": -r["mean_test_brier"][i],
            "parametros": json.dumps(busca.best_params_),
        })
        buscas[nome] = busca
    tabela = pd.DataFrame(linhas).sort_values("auc_pr", ascending=False, ignore_index=True)
    return tabela, buscas[tabela.loc[0, "modelo"]].best_estimator_


def definir_cortes(modelo, treino: pd.DataFrame) -> dict:
    """Cortes tirados de previsões fora da dobra, sem olhar o teste."""
    prob = cross_val_predict(clone(modelo), treino[ENTRADAS], treino["alvo"],
                             cv=dobras(treino), method="predict_proba")[:, 1]
    return {"alta": float(np.quantile(prob, 0.80)), "media": float(np.quantile(prob, 0.50))}


def avaliar_no_tempo(modelo, dados: pd.DataFrame) -> dict:
    """Teste de estresse: treina nos 80% mais antigos e avalia nos 20% mais recentes."""
    n = int(len(dados) * 0.8)
    passado, futuro = dados.iloc[:n].reset_index(drop=True), dados.iloc[n:]
    corte = definir_cortes(modelo, passado)["alta"]
    ajustado = clone(modelo).fit(passado[ENTRADAS], passado["alvo"])
    resultado = metricas(futuro["alvo"], ajustado.predict_proba(futuro[ENTRADAS])[:, 1], corte)
    return {"taxa_passado": float(passado["alvo"].mean()), **resultado}


def salvar_figuras(teste, prob, corte, importancia) -> None:
    pasta = SAIDA / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        teste["alvo"], prob >= corte, display_labels=["Não", "Sim"],
        values_format="d", cmap="Blues", colorbar=False, ax=ax)
    ax.set(title="Matriz de confusão (teste)", xlabel="Prioridade alta?", ylabel="Contratou?")
    fig.tight_layout()
    fig.savefig(pasta / "matriz_confusao.png", dpi=120)
    plt.close(fig)

    real, previsto = calibration_curve(teste["alvo"], prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    limite = max(real.max(), previsto.max()) * 1.1
    ax.plot([0, limite], [0, limite], "--", color="gray", label="ideal")
    ax.plot(previsto, real, "o-", label="modelo")
    ax.set(title="Calibração (teste)", xlabel="Probabilidade prevista", ylabel="Adesão observada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "calibracao.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    importancia.sort_values().plot.barh(ax=ax)
    ax.set(title="Importância por permutação (teste)", xlabel="Queda na AUC-PR", ylabel="")
    fig.tight_layout()
    fig.savefig(pasta / "importancia.png", dpi=120)
    plt.close(fig)


def executar() -> dict:
    dados = carregar()
    treino, teste = separar(dados)

    comparacao, modelo = comparar(treino)
    cortes = definir_cortes(modelo, treino)

    # O teste só é usado daqui para baixo, com modelo e cortes já definidos
    prob = modelo.predict_proba(teste[ENTRADAS])[:, 1]
    erros = erros_por_segmento(teste, prob, cortes["alta"])
    perm = permutation_importance(modelo, teste[ENTRADAS], teste["alvo"],
                                  scoring="average_precision", n_repeats=5, random_state=SEMENTE)
    importancia = pd.Series(perm.importances_mean, index=ENTRADAS)
    faixa = np.select([prob >= cortes["alta"], prob >= cortes["media"]], ["Alta", "Média"], "Baixa")
    resumo = {
        "modelo": comparacao.loc[0, "modelo"],
        "parametros": json.loads(comparacao.loc[0, "parametros"]),
        "cortes": cortes,
        "taxa_treino": float(treino["alvo"].mean()),
        "linhas_treino": len(treino),
        "teste": metricas(teste["alvo"], prob, cortes["alta"]),
        "teste_corte_050": metricas(teste["alvo"], prob, 0.5),
        "faixas": {f: {"contatos": int((faixa == f).sum()),
                       "adesao": float(teste.loc[faixa == f, "alvo"].mean())}
                   for f in ["Alta", "Média", "Baixa"]},
        "tempo": avaliar_no_tempo(modelo, dados),
    }

    SAIDA.mkdir(exist_ok=True)
    (RAIZ / "modelos").mkdir(exist_ok=True)
    joblib.dump({"pipeline": modelo, "cortes": cortes, "taxa_treino": resumo["taxa_treino"],
                 "sklearn": sklearn.__version__}, RAIZ / "modelos" / "modelo.joblib", compress=3)
    (SAIDA / "metricas.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    tabela_importancia = importancia.rename_axis("variavel").rename("queda_auc_pr").reset_index()
    for nome, tabela in [("comparacao_modelos", comparacao), ("erros_por_segmento", erros),
                         ("importancia", tabela_importancia)]:
        tabela.to_csv(SAIDA / f"{nome}.csv", index=False, float_format="%.6f")
    salvar_figuras(teste, prob, cortes["alta"], importancia)
    return {"resumo": resumo, "comparacao": comparacao, "erros": erros, "importancia": importancia}


if __name__ == "__main__":
    r = executar()
    print(r["comparacao"].to_string(index=False))
    print(json.dumps(r["resumo"], indent=2, ensure_ascii=False))
