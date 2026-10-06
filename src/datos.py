"""Carga y validación del dataset de reseñas.

El dataset lo produce el proyecto hermano en R (`TFG_LEROY_MERLIN_CORREGIDO`),
que es quien define las tres uniones entre las cuatro fuentes de Leroy Merlin.
Aquí NO se reimplementan esas uniones a propósito: una segunda implementación
acabaría divergiendo de la primera sin que nadie se diera cuenta, y entonces
las dos versiones del proyecto dejarían de ser comparables.

Para regenerarlo:
    Rscript --vanilla R/run_all.R      # escribe data_interim/databoceto.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Atributos técnicos del producto. Son los mismos once predictores de la
# versión R, y el orden importa sólo para que los informes sean estables.
ATRIBUTOS = [
    "MARCA", "FONDO REDUCIDO", "RIMLESS", "TAPA AMORTIGUADA", "SALIDA DUAL",
    "TAPA EXTRAIBLE", "COLOR", "GARANTIA", "PRECIO", "TOP 2500", "MARCA PROPIA",
]
NUMERICAS = ["PRECIO"]
CATEGORICAS = [c for c in ATRIBUTOS if c not in NUMERICAS]

COL_TEXTO = "COMENTARIO"
COL_TITULO = "TITULO"
COL_GRUPO = "Descripción"      # el producto: la unidad que define los grupos
COL_OBJETIVO = "QUEJAS"

_POR_DEFECTO = (
    r"C:\Users\Usuario\Desktop\Claude\TFG_LEROY_MERLIN_CORREGIDO"
    r"\data_interim\databoceto.csv"
)


def ruta_datos() -> Path:
    """Ruta al CSV. Configurable con la variable de entorno RESENAS_DATOS."""
    return Path(os.environ.get("RESENAS_DATOS", _POR_DEFECTO))


def cargar(ruta: str | Path | None = None) -> pd.DataFrame:
    """Carga el dataset y comprueba que es el que esperamos.

    Las comprobaciones no son decorativas: si el CSV cambia de forma (otra
    versión del pipeline R, otro filtrado), los resultados de este proyecto
    dejarían de ser comparables con los de la versión R y conviene enterarse
    aquí y no tres gráficos más tarde.
    """
    ruta = Path(ruta) if ruta is not None else ruta_datos()
    if not ruta.exists():
        raise FileNotFoundError(
            f"No está {ruta}.\n"
            "Genéralo con:  Rscript --vanilla R/run_all.R  en el proyecto R, "
            "o apunta la variable de entorno RESENAS_DATOS al fichero."
        )

    df = pd.read_csv(ruta, encoding="utf-8")

    faltan = [c for c in (ATRIBUTOS + [COL_TEXTO, COL_GRUPO, COL_OBJETIVO]) if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta.name}: {faltan}")

    # El objetivo: SI = hubo queja (reseña de 1 a 4 estrellas).
    df[COL_OBJETIVO] = df[COL_OBJETIVO].astype(str).str.strip().str.upper()
    valores = set(df[COL_OBJETIVO].unique())
    if not valores <= {"SI", "NO"}:
        raise ValueError(f"{COL_OBJETIVO} tiene valores inesperados: {valores}")

    df[COL_TEXTO] = df[COL_TEXTO].fillna("").astype(str)
    df[COL_TITULO] = df.get(COL_TITULO, "").fillna("").astype(str)
    df["PRECIO"] = pd.to_numeric(df["PRECIO"], errors="coerce")

    for c in CATEGORICAS:
        df[c] = df[c].astype(str).str.strip().str.upper()

    return df


def objetivo_binario(df: pd.DataFrame) -> pd.Series:
    """1 = hubo queja. La clase positiva es la queja porque es la que interesa."""
    return (df[COL_OBJETIVO] == "SI").astype(int)


def resumen(df: pd.DataFrame) -> str:
    y = objetivo_binario(df)
    n_prod = df[COL_GRUPO].nunique()
    vectores = df[ATRIBUTOS].astype(str).agg("|".join, axis=1).nunique()
    base = max(y.mean(), 1 - y.mean())
    return (
        f"{len(df)} reseñas · {n_prod} productos · {len(df)/n_prod:.1f} por producto\n"
        f"{vectores} vectores de atributos únicos  "
        f"(cada uno se repite {len(df)/vectores:.1f} veces)\n"
        f"quejas: {y.sum()} ({y.mean():.1%})  ·  baseline: {base:.4f}"
    )
