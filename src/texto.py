"""Procesado de texto en español.

Decisión técnica: no se usa spaCy ni lematización.

El motivo no es la pereza. El corpus son 884 reseñas de 97 caracteres de
media, escritas por clientes con faltas de ortografía ("guestado", "vater")
y sin tildes la mitad de las veces. Un lematizador entrenado en texto
normativo acierta poco ahí, y añade una dependencia de ~50 MB que impide
que alguien clone el repositorio y lo ejecute.

En su lugar se combinan dos vistas del texto:
  - n-gramas de PALABRA (1-2): capturan expresiones como "mala calidad"
  - n-gramas de CARACTER (3-5): absorben la morfología del español
    (roto/rota/rotos) y las faltas, sin necesidad de lematizar

Es la aproximación que mejor aguanta texto de usuario real, y se puede
defender en una entrevista mejor que "usé spaCy porque es lo que se usa".
"""

from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# Stopwords del español. Lista corta y revisada a mano: las listas largas
# estándar incluyen "no", "nada", "poco", "nunca" y "muy", que en análisis
# de opiniones son justo las palabras que marcan la diferencia entre
# "buena calidad" y "no es de buena calidad".
STOPWORDS_ES = {
    "a", "al", "algo", "ante", "antes", "aqui", "asi", "aun", "aunque", "cada",
    "como", "con", "cual", "cuando", "de", "del", "desde", "donde", "dos", "el",
    "ella", "ellos", "en", "entre", "era", "eran", "es", "esa", "ese", "eso",
    "esta", "estaba", "estan", "este", "esto", "estos", "fue", "ha", "habia",
    "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "me", "mi",
    "mis", "mucho", "nos", "nuestro", "o", "os", "otra", "otro", "para", "pero",
    "por", "porque", "que", "se", "segun", "ser", "si", "sin", "sobre", "son",
    "su", "sus", "tambien", "te", "tiene", "tienen", "todo", "tu", "un", "una",
    "unas", "uno", "unos", "ya", "yo",
}
# Deliberadamente FUERA de la lista: no, nada, nunca, poco, muy, mas, menos,
# bien, mal. Son portadoras de polaridad.

_NO_LETRA = re.compile(r"[^a-zñ ]+")
_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sólo letras.

    Se quitan las tildes porque los clientes las ponen de forma inconsistente
    ("rapido" y "rápido" son la misma palabra para este análisis). La ñ se
    conserva: en español distingue palabras, no es un acento.
    """
    if not isinstance(texto, str):
        return ""
    t = texto.lower().replace("ñ", "\x00")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("\x00", "ñ")
    t = _NO_LETRA.sub(" ", t)
    return _ESPACIOS.sub(" ", t).strip()


def tokenizar(texto: str, quitar_stopwords: bool = True) -> list[str]:
    palabras = normalizar(texto).split()
    if quitar_stopwords:
        palabras = [p for p in palabras if p not in STOPWORDS_ES and len(p) > 2]
    return palabras


def construir_vectorizador(min_df: int = 3) -> FeatureUnion:
    """Palabra (1-2) + carácter (3-5), unidos.

    min_df=3 descarta lo que aparece en menos de tres reseñas: con 884
    documentos, un término que sale dos veces no sostiene una conclusión y
    sólo añade ruido y riesgo de sobreajuste.
    """
    return FeatureUnion([
        ("palabra", TfidfVectorizer(
            preprocessor=normalizar,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
            stop_words=sorted(STOPWORDS_ES),
        )),
        ("caracter", TfidfVectorizer(
            preprocessor=normalizar,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            sublinear_tf=True,
        )),
    ])


def combinar_titulo_comentario(df, col_titulo="TITULO", col_texto="COMENTARIO"):
    """El título es corto pero muy informativo ('Decepcionada', 'Excelente').

    Se concatena en vez de usarse aparte porque con 884 filas no hay datos
    para permitirse dos bloques de features que dicen casi lo mismo.
    """
    return (df[col_titulo].fillna("") + ". " + df[col_texto].fillna("")).str.strip()
