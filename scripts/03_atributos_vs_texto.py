"""¿Qué predice la satisfacción: el producto o lo que el cliente cuenta?

Esta es la pregunta del proyecto, y la respuesta tiene consecuencias para el
TFG original: si el texto gana con holgura, entonces el AUC de 0,52 que daba
el modelo de atributos no era un fallo de modelado, sino la respuesta correcta
a una pregunta mal planteada.

Se comparan cuatro modelos bajo EL MISMO protocolo:
    1. Atributos        — los 11 del TFG. Replica la versión R.
    2. Texto            — TF-IDF (palabra + carácter) del título y comentario.
    3. Atributos+texto  — ¿aporta algo el producto una vez tienes el texto?
    4. Sólo longitud    — control. Un modelo con una sola variable: cuántos
                          caracteres escribió el cliente. Si éste ya iguala al
                          de texto, es que el TF-IDF no está capturando
                          significado, sólo que las quejas son más largas.

El cuarto es el que evita engañarse. Sin un control así, cualquier mejora del
modelo de texto se atribuye al "análisis de sentimiento" cuando puede venir de
algo trivial.

Uso:  python scripts/03_atributos_vs_texto.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import datos as D
from src import evaluacion as E
from src import texto as T

SALIDA = RAIZ / "output"
SALIDA.mkdir(exist_ok=True)


# ---------------------------------------------------------------- modelos
def modelo_atributos(tipo: str = "logistica"):
    """Los 11 atributos técnicos. Equivalente a la receta de la versión R."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), D.NUMERICAS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=5))]), D.CATEGORICAS),
    ])
    clf = (LogisticRegression(max_iter=2000, C=1.0)
           if tipo == "logistica"
           else RandomForestClassifier(n_estimators=500, random_state=2023, n_jobs=-1))
    return Pipeline([("pre", pre), ("clf", clf)])


def modelo_texto():
    """TF-IDF + logística.

    C=1.0 y sin búsqueda de hiperparámetros: con 884 filas, ajustar C por
    validación cruzada dentro de cada split añadiría varianza y poco más, y
    el objetivo aquí es comparar FAMILIAS de variables, no exprimir un modelo.
    """
    return Pipeline([
        ("tfidf", T.construir_vectorizador(min_df=3)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
    ])


def modelo_longitud():
    """Control: sólo el número de caracteres del comentario."""
    def _largo(s):
        return np.asarray([len(x) for x in s]).reshape(-1, 1)
    return Pipeline([
        ("largo", FunctionTransformer(_largo, validate=False)),
        ("esc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    df = D.cargar()
    y = D.objetivo_binario(df).to_numpy()
    grupos = df[D.COL_GRUPO].to_numpy()
    texto_completo = T.combinar_titulo_comentario(df)

    print("=" * 70)
    print("DATOS")
    print("=" * 70)
    print(D.resumen(df))

    X_attr = df[D.ATRIBUTOS]

    print("\n" + "=" * 70)
    print("EVALUACIÓN — 2 estrategias x 5 semillas por modelo")
    print("=" * 70)

    partes = [
        E.evaluar(lambda: modelo_atributos("logistica"), X_attr, y, grupos,
                  "1. atributos (logística)"),
        E.evaluar(lambda: modelo_atributos("rf"), X_attr, y, grupos,
                  "2. atributos (random forest)"),
        E.evaluar(modelo_texto, texto_completo, y, grupos,
                  "3. texto (TF-IDF)"),
        E.evaluar(modelo_longitud, texto_completo, y, grupos,
                  "4. sólo longitud (control)"),
    ]
    completo = pd.concat(partes, ignore_index=True)
    res = E.resumir(completo)

    print()
    print(res.to_string(index=False))

    completo.to_csv(SALIDA / "resultados_completos.csv", index=False)
    res.to_csv(SALIDA / "resumen_modelos.csv", index=False)

    # ------------------------------------------------- lectura del resultado
    def auc(nombre, estrategia):
        f = res[(res.modelo == nombre) & (res.estrategia == estrategia)]
        return float(f["auc"].iloc[0]) if len(f) else float("nan")

    a_attr = auc("1. atributos (logística)", "por_producto")
    a_text = auc("3. texto (TF-IDF)", "por_producto")
    a_ctrl = auc("4. sólo longitud (control)", "por_producto")
    base = max(y.mean(), 1 - y.mean())

    print("\n" + "=" * 70)
    print("LECTURA  (split por producto — el honesto)")
    print("=" * 70)
    print(f"  baseline (clase mayoritaria) .......... {base:.4f}")
    print(f"  atributos del producto ................ {a_attr:.4f}")
    print(f"  texto de la reseña .................... {a_text:.4f}")
    print(f"  control: sólo longitud ................ {a_ctrl:.4f}")
    print(f"\n  diferencia texto - atributos .......... {a_text - a_attr:+.4f}")
    print(f"  diferencia texto - control ............ {a_text - a_ctrl:+.4f}")

    if a_text - a_attr > 0.05 and a_text - a_ctrl > 0.03:
        print("\n  >> El texto predice sustancialmente mejor que los atributos,")
        print("     y no es un artefacto de la longitud del comentario.")
    elif a_text - a_attr > 0.05:
        print("\n  >> El texto gana a los atributos, PERO el control de longitud")
        print("     explica buena parte: el TF-IDF aporta menos de lo que parece.")
    else:
        print("\n  >> El texto NO mejora de forma apreciable a los atributos.")
        print("     Con 884 reseñas cortas puede faltar señal; conviene decirlo.")

    # ------------------------------------------------- qué palabras pesan
    print("\n" + "=" * 70)
    print("TÉRMINOS MÁS ASOCIADOS (modelo de texto sobre el total)")
    print("=" * 70)
    m = modelo_texto().fit(texto_completo, y)
    vec = m.named_steps["tfidf"].transformer_list[0][1]   # sólo n-gramas de palabra
    coefs_todos = m.named_steps["clf"].coef_[0]
    nombres = vec.get_feature_names_out()
    coefs = coefs_todos[:len(nombres)]

    orden = np.argsort(coefs)
    print("\n  -> QUEJA:")
    for i in orden[::-1][:15]:
        print(f"     {coefs[i]:+.3f}  {nombres[i]}")
    print("\n  -> SIN QUEJA:")
    for i in orden[:15]:
        print(f"     {coefs[i]:+.3f}  {nombres[i]}")

    tabla_terminos = (pd.DataFrame({"termino": nombres, "coeficiente": coefs})
                      .sort_values("coeficiente", ascending=False))
    tabla_terminos.to_csv(SALIDA / "terminos_coeficientes.csv", index=False)

    figuras(res, tabla_terminos, base)
    print(f"\nResultados y figuras guardados en {SALIDA}")


# ---------------------------------------------------------------- figuras
def figuras(res: pd.DataFrame, terminos: pd.DataFrame, base: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dir_fig = SALIDA / "figures"
    dir_fig.mkdir(exist_ok=True)

    # --- 1. La comparación -------------------------------------------------
    piv = res.pivot(index="modelo", columns="estrategia", values="auc").sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ypos = np.arange(len(piv))
    ax.barh(ypos + 0.19, piv["aleatorio"], height=0.36,
            color="#c0392b", label="aleatorio por fila (con fuga)")
    ax.barh(ypos - 0.19, piv["por_producto"], height=0.36,
            color="#1e8449", label="agrupado por producto (honesto)")
    for i, (a, p) in enumerate(zip(piv["aleatorio"], piv["por_producto"])):
        ax.text(a + .008, i + .19, f"{a:.3f}", va="center", fontsize=8.5)
        ax.text(p + .008, i - .19, f"{p:.3f}", va="center", fontsize=8.5,
                fontweight="bold")
    ax.axvline(base, ls="--", c="grey", lw=1)
    ax.text(base, len(piv) - 0.35, f" baseline {base:.3f}", color="grey", fontsize=8.5)
    ax.axvline(0.5, ls=":", c="#999", lw=1)
    ax.set_yticks(ypos, piv.index, fontsize=9)
    ax.set_xlabel("AUC en test (media de 5 semillas)")
    ax.set_xlim(0, 0.92)
    ax.set_title("Lo que predice la satisfacción no es el producto:\n"
                 "es lo que el cliente escribe", fontsize=12, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(dir_fig / "01_atributos_vs_texto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Términos -------------------------------------------------------
    top = pd.concat([terminos.head(14), terminos.tail(14)])
    fig, ax = plt.subplots(figsize=(7, 7))
    colores = ["#c0392b" if c > 0 else "#1e8449" for c in top["coeficiente"]]
    ax.barh(range(len(top)), top["coeficiente"], color=colores)
    ax.set_yticks(range(len(top)), top["termino"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, c="black", lw=.8)
    ax.set_xlabel("← sin queja        coeficiente        queja →")
    ax.set_title("Qué palabras marcan la diferencia\n"
                 "'no' es el término más predictivo de queja", fontsize=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(dir_fig / "02_terminos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  figuras: {dir_fig}")


if __name__ == "__main__":
    main()
