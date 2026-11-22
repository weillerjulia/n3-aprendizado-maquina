"""Métricas e análise de erros."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)


def metricas(real, prob, corte: float) -> dict:
    real, prob = np.asarray(real), np.asarray(prob)
    previsto = prob >= corte
    vn, fp, fn, vp = confusion_matrix(real, previsto, labels=[0, 1]).ravel()
    precisao = precision_score(real, previsto, zero_division=0)
    return {
        "contatos": len(real),
        "taxa_real": float(real.mean()),
        "probabilidade_media": float(prob.mean()),
        "auc_pr": float(average_precision_score(real, prob)),
        "auc_roc": float(roc_auc_score(real, prob)),
        "brier": float(brier_score_loss(real, prob)),
        "selecionados": float(previsto.mean()),
        "precisao": float(precisao),
        "revocacao": float(recall_score(real, previsto, zero_division=0)),
        "lift": float(precisao / real.mean()),
        "vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp),
    }


def erros_por_segmento(dados: pd.DataFrame, prob, corte: float) -> pd.DataFrame:
    """Onde o modelo deixa de achar quem contrata e onde gera ligações em vão."""
    base = dados.assign(
        previsto=np.asarray(prob) >= corte,
        idade=pd.cut(dados["age"], [0, 25, 40, 60, 120],
                     labels=["até 25", "26 a 40", "41 a 60", "mais de 60"]).astype(str),
    )
    linhas = []
    for coluna in ["idade", "job", "contact", "poutcome"]:
        for valor, g in base.groupby(coluna):
            positivos = int(g["alvo"].sum())
            achados = int((g["previsto"] & (g["alvo"] == 1)).sum())
            selecionados = int(g["previsto"].sum())
            linhas.append({
                "segmento": coluna, "valor": valor, "contatos": len(g), "adesoes": positivos,
                "falsos_negativos": positivos - achados,
                "falsos_positivos": selecionados - achados,
                "revocacao": achados / positivos if positivos else np.nan,
                "precisao": achados / selecionados if selecionados else np.nan,
            })
    return pd.DataFrame(linhas)
