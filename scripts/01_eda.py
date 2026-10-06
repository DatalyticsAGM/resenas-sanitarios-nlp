"""Exploración: qué forma tienen 884 reseñas de 26 inodoros.

El objetivo de este script no es decorar el repositorio con gráficos, sino
dejar documentadas las tres cosas que condicionan todo lo demás:

  1. Cuántas reseñas hay por producto (de ahí viene la fuga).
  2. Cómo se reparten las estrellas, y por qué la frontera está en el 5.
  3. Que la etiqueta tiene ruido: hay reseñas de 4 estrellas con texto
     claramente positivo, y el TFG las cuenta como queja.

Uso:  python scripts/01_eda.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import datos as D

SALIDA = RAIZ / "output"
DIR_FIG = SALIDA / "figures"
DIR_FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = D.cargar()
    print("=" * 70)
    print("EXPLORACIÓN")
    print("=" * 70)
    print(D.resumen(df), "\n")

    # --- 1. Estrellas ------------------------------------------------------
    estrellas = pd.to_numeric(df["VALORACION RESEÑA"], errors="coerce")
    conteo = estrellas.value_counts().sort_index()
    print("Reparto de estrellas")
    for k, v in conteo.items():
        print(f"  {int(k)}★  {v:4d}  {'█' * int(v / 6)}")
    print(f"\n  5★ = {conteo.get(5, 0)} ({conteo.get(5,0)/len(df):.1%}) -> 'sin queja'")
    print(f"  1-4★ = {len(df)-conteo.get(5,0)} -> 'queja'")
    print("  El corte en 5 es del autor del TFG, no del enunciado: en la escala")
    print("  de Leroy Merlin cualquier nota por debajo de 5 señala algo que falla.")

    # --- 2. Reseñas por producto ------------------------------------------
    por_prod = df.groupby(D.COL_GRUPO).size().sort_values(ascending=False)
    print(f"\nReseñas por producto: mediana {por_prod.median():.0f} · "
          f"min {por_prod.min()} · max {por_prod.max()}")
    print(f"  Los 5 productos más reseñados concentran "
          f"{por_prod.head(5).sum()/len(df):.1%} de las reseñas.")

    # --- 3. El ruido de la etiqueta ---------------------------------------
    pos = ["bien", "buen", "buena", "perfecto", "excelente", "genial",
           "encanta", "recomiendo", "contento", "contenta", "gusta"]
    cuatro = df[estrellas == 4]
    patron = "|".join(pos)
    tono_pos = cuatro["COMENTARIO"].str.lower().str.contains(patron, na=False)
    print(f"\nRuido de la etiqueta: de {len(cuatro)} reseñas de 4★ "
          f"(contadas como QUEJA), {tono_pos.sum()} ({tono_pos.mean():.0%}) "
          f"contienen vocabulario positivo.")
    print("  Es un techo para cualquier modelo: parte de las 'quejas' no lo son.")

    # --- 4. Precio ---------------------------------------------------------
    q = D.objetivo_binario(df)
    print(f"\nPrecio medio  con queja: {df.loc[q==1,'PRECIO'].mean():7.2f} € · "
          f"sin queja: {df.loc[q==0,'PRECIO'].mean():7.2f} €")

    # --- Figuras -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(conteo.index, conteo.values,
                color=["#c0392b"] * 4 + ["#1e8449"])
    axes[0].set_title("Estrellas por reseña", loc="left", fontsize=11)
    axes[0].set_xlabel("valoración"); axes[0].set_ylabel("reseñas")

    axes[1].hist(por_prod.values, bins=12, color="#5d6d7e")
    axes[1].set_title(f"Reseñas por producto (n={len(por_prod)})",
                      loc="left", fontsize=11)
    axes[1].set_xlabel("reseñas"); axes[1].set_ylabel("productos")

    largo = df["COMENTARIO"].str.len()
    datos_box = [largo[estrellas == k].values for k in sorted(conteo.index)]
    axes[2].boxplot(datos_box, tick_labels=[int(k) for k in sorted(conteo.index)],
                    showfliers=False)
    axes[2].set_title("Longitud del comentario", loc="left", fontsize=11)
    axes[2].set_xlabel("estrellas"); axes[2].set_ylabel("caracteres")

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Las reseñas insatisfechas son más largas: el cliente explica el problema",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DIR_FIG / "00_eda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura en {DIR_FIG / '00_eda.png'}")


if __name__ == "__main__":
    main()
