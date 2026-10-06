"""Protocolo de evaluación común.

Todo el valor de este proyecto depende de que los modelos de atributos y los
de texto se evalúen EXACTAMENTE igual. Si cada uno usa su propio split, la
comparación no significa nada y volveríamos al error que este trabajo
denuncia. Por eso el protocolo vive aquí y no dentro de cada script.

Dos estrategias, las mismas que en la versión R:

  "aleatorio"  — divide por fila, ignorando el producto.
                 Es lo que hacía el TFG original. Produce fuga cuando los
                 predictores son atributos del producto, porque el mismo
                 vector cae a los dos lados.

  "por_producto" — divide por producto: ningún producto aparece en train y
                 en test a la vez. Es el único honesto para los atributos.

Las cinco semillas son las mismas que en R (11, 42, 77, 2017, 123) para que
las tablas de los dos proyectos se puedan poner una al lado de la otra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

SEMILLAS = [11, 42, 77, 2017, 123]
PROP_ENTRENAMIENTO = 0.80


def particiones(y, grupos, estrategia: str, semilla: int):
    """Devuelve (idx_train, idx_test) según la estrategia pedida."""
    n = len(y)
    if estrategia == "aleatorio":
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=PROP_ENTRENAMIENTO, random_state=semilla)
        return next(sss.split(np.zeros(n), y))
    if estrategia == "por_producto":
        # Nota: GroupShuffleSplit reparte GRUPOS, así que la proporción de
        # filas resultante no es exactamente 0,80 (los productos no tienen
        # todos el mismo número de reseñas). rsample en R hace lo mismo.
        gss = GroupShuffleSplit(
            n_splits=1, train_size=PROP_ENTRENAMIENTO, random_state=semilla)
        return next(gss.split(np.zeros(n), y, groups=grupos))
    raise ValueError(f"Estrategia desconocida: {estrategia}")


def metricas(y_true, y_prob, umbral: float = 0.5) -> dict:
    """AUC, accuracy, sensibilidad y especificidad.

    La clase positiva es la QUEJA. Sensibilidad = qué porcentaje de las
    quejas detecta el modelo, que es lo que le importaría a la tienda.
    """
    y_pred = (y_prob >= umbral).astype(int)
    # Si en un test sólo hay una clase, el AUC no está definido. Pasa cuando
    # el split por producto deja un test pequeño y homogéneo: hay que
    # reportarlo como NaN, no inventar un 0,5.
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensibilidad": tp / (tp + fn) if (tp + fn) else np.nan,
        "especificidad": tn / (tn + fp) if (tn + fp) else np.nan,
    }


def solapamiento(grupos, idx_train, idx_test) -> float:
    """% de filas de test cuyo producto ya se vio en entrenamiento.

    Es la medida directa de la fuga. Con split por producto debe dar 0,0.
    """
    g = np.asarray(grupos)
    vistos = set(g[idx_train])
    return 100.0 * np.mean([x in vistos for x in g[idx_test]])


def evaluar(construir_modelo, X, y, grupos, nombre: str,
            estrategias=("aleatorio", "por_producto"),
            semillas=None) -> pd.DataFrame:
    """Entrena y evalúa un modelo bajo cada estrategia y cada semilla.

    `construir_modelo` es una función sin argumentos que devuelve un
    estimador nuevo. Se pide una FÁBRICA y no un modelo ya construido para
    garantizar que cada repetición entrena desde cero: reutilizar el mismo
    objeto arrastraría el ajuste anterior.
    """
    semillas = list(semillas or SEMILLAS)
    filas = []
    for estrategia in estrategias:
        for s in semillas:
            tr, te = particiones(y, grupos, estrategia, s)
            modelo = construir_modelo()
            Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            Xte = X.iloc[te] if hasattr(X, "iloc") else X[te]
            modelo.fit(Xtr, np.asarray(y)[tr])
            prob = modelo.predict_proba(Xte)[:, 1]
            m = metricas(np.asarray(y)[te], prob)
            filas.append({
                "modelo": nombre,
                "estrategia": estrategia,
                "semilla": s,
                "n_train": len(tr),
                "n_test": len(te),
                "solape_pct": round(solapamiento(grupos, tr, te), 1),
                **m,
            })
    return pd.DataFrame(filas)


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Media sobre las semillas, que es lo que se reporta."""
    cols = ["auc", "accuracy", "sensibilidad", "especificidad", "solape_pct"]
    out = (df.groupby(["modelo", "estrategia"], sort=False)[cols]
             .mean().round(4).reset_index())
    return out
