"""Qué dicen las reseñas, antes de modelar nada.

El TFG original dedicó 487 líneas de R a esto (udpipe, bigramas, grafos de
co-ocurrencia). Aquí se reconstruye la parte que sostiene conclusiones, con
una diferencia: se usa log-odds en vez de frecuencia bruta.

Por qué importa: las palabras más FRECUENTES en las quejas son las mismas que
en los elogios ("calidad", "precio", "producto"), porque son el vocabulario
del dominio. Ordenar por frecuencia produce una nube de palabras bonita que
no dice nada. El log-odds mide qué términos son DESPROPORCIONADAMENTE
frecuentes en una clase, que es la pregunta que de verdad interesa.

Uso:  python scripts/02_nlp_descriptivo.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import datos as D
from src import texto as T

SALIDA = RAIZ / "output"
DIR_FIG = SALIDA / "figures"
DIR_FIG.mkdir(parents=True, exist_ok=True)


def log_odds(conteo_a: Counter, conteo_b: Counter, min_total: int = 5) -> pd.DataFrame:
    """Log-odds con suavizado de Laplace.

    Positivo = más propio de A (queja). Negativo = más propio de B.
    `min_total` evita que un término que aparece 2 veces encabece la tabla
    sólo por salir las dos en la misma clase.
    """
    vocab = set(conteo_a) | set(conteo_b)
    na, nb = sum(conteo_a.values()), sum(conteo_b.values())
    filas = []
    for t in vocab:
        a, b = conteo_a.get(t, 0), conteo_b.get(t, 0)
        if a + b < min_total:
            continue
        pa = (a + 1) / (na + len(vocab))
        pb = (b + 1) / (nb + len(vocab))
        filas.append({"termino": t, "n_queja": a, "n_sin_queja": b,
                      "log_odds": np.log(pa / pb)})
    return pd.DataFrame(filas).sort_values("log_odds", ascending=False)


def contar(serie, n: int = 1) -> Counter:
    c = Counter()
    for txt in serie:
        tokens = T.tokenizar(txt)
        if n == 1:
            c.update(tokens)
        else:
            c.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return c


def main() -> None:
    df = D.cargar()
    y = D.objetivo_binario(df)
    txt = T.combinar_titulo_comentario(df)
    queja, sin_queja = txt[y == 1], txt[y == 0]

    print("=" * 70)
    print("ANÁLISIS DESCRIPTIVO DEL TEXTO")
    print("=" * 70)
    print(f"{len(queja)} reseñas con queja · {len(sin_queja)} sin queja\n")

    # --- Por qué la frecuencia bruta no sirve ------------------------------
    c_q, c_s = contar(queja), contar(sin_queja)
    print("Las 8 palabras más frecuentes en cada clase:")
    print(f"  con queja: {', '.join(t for t, _ in c_q.most_common(8))}")
    print(f"  sin queja: {', '.join(t for t, _ in c_s.most_common(8))}")
    comunes = set(t for t, _ in c_q.most_common(8)) & set(t for t, _ in c_s.most_common(8))
    print(f"  -> {len(comunes)} de 8 coinciden: {', '.join(sorted(comunes))}")
    print("     Por eso no se ordena por frecuencia.\n")

    # --- Log-odds, unigramas ----------------------------------------------
    uni = log_odds(c_q, c_s)
    print("Términos propios de la QUEJA (log-odds):")
    for _, r in uni.head(12).iterrows():
        print(f"  {r.log_odds:+.2f}  {r.termino:<18} ({int(r.n_queja)} vs {int(r.n_sin_queja)})")
    print("\nTérminos propios de SIN QUEJA:")
    for _, r in uni.tail(12).iloc[::-1].iterrows():
        print(f"  {r.log_odds:+.2f}  {r.termino:<18} ({int(r.n_queja)} vs {int(r.n_sin_queja)})")

    # --- Bigramas ----------------------------------------------------------
    bi = log_odds(contar(queja, 2), contar(sin_queja, 2), min_total=4)
    print("\nBigramas propios de la QUEJA:")
    for _, r in bi.head(10).iterrows():
        print(f"  {r.log_odds:+.2f}  {r.termino}")

    uni.to_csv(SALIDA / "terminos_log_odds.csv", index=False)
    bi.to_csv(SALIDA / "bigramas_log_odds.csv", index=False)

    # --- Figura ------------------------------------------------------------
    top = pd.concat([uni.head(15), uni.tail(15)])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.barh(range(len(top)), top["log_odds"],
            color=["#c0392b" if v > 0 else "#1e8449" for v in top["log_odds"]])
    ax.set_yticks(range(len(top)), top["termino"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, c="black", lw=.8)
    ax.set_xlabel("← propio de reseñas sin queja     log-odds     propio de quejas →")
    ax.set_title("Vocabulario característico de cada clase\n"
                 "ordenado por log-odds, no por frecuencia", fontsize=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "03_log_odds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura en {DIR_FIG / '03_log_odds.png'}")


if __name__ == "__main__":
    main()
