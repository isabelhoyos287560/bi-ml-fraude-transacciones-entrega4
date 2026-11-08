"""Funciones compartidas por los notebooks del Evento 4 (fraude IEEE-CIS).

Regla de oro contra la fuga de datos: todo lo que "aprende" de los datos
(columnas a eliminar, imputación, escalado, codificación) se ajusta SOLO con
el conjunto de entrenamiento y luego se aplica a prueba.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "isFraud"
COLS_EXCLUIDAS = ["TransactionID", "TransactionDT", TARGET]


def cargar_datos(data_dir):
    """Une transacciones e identidad por TransactionID (la identidad no existe para todas)."""
    data_dir = Path(data_dir)
    trans = pd.read_csv(data_dir / "train_transaction.csv")
    ident = pd.read_csv(data_dir / "train_identity.csv")
    return trans.merge(ident, on="TransactionID", how="left")


def muestrear_y_dividir(df, n_muestra, seed=42, frac_train=0.7):
    """Muestra aleatoria (conserva la proporción de fraude) y división POR TIEMPO.

    Se entrena con lo más antiguo y se prueba con lo más reciente, como pasaría
    en producción. Un corte aleatorio mezclaría futuro y pasado.
    """
    if n_muestra and n_muestra < len(df):
        df = df.sample(n_muestra, random_state=seed)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    corte = int(len(df) * frac_train)
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def agregar_features(df):
    """Variables derivadas que no aprenden nada de los datos (sin riesgo de fuga)."""
    df = df.copy()
    df["hora"] = (df["TransactionDT"] // 3600) % 24
    df["dia_semana"] = (df["TransactionDT"] // 86400) % 7
    df["log_monto"] = np.log1p(df["TransactionAmt"])
    return df


def seleccionar_columnas(train, umbral_nulos=0.90, umbral_corr=0.95, seed=42):
    """Decide qué columnas conservar usando SOLO el train.

    Elimina: columnas casi vacías, constantes y numéricas muy correlacionadas.
    Devuelve (columnas_numericas, columnas_categoricas, informe).
    """
    X = train.drop(columns=[c for c in COLS_EXCLUIDAS if c in train.columns])
    informe = {"inicial": X.shape[1]}

    nulos = X.isna().mean()
    X = X.loc[:, nulos <= umbral_nulos]
    informe["quitadas_por_nulos"] = int((nulos > umbral_nulos).sum())

    constantes = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constantes)
    informe["quitadas_por_constantes"] = len(constantes)

    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]

    muestra = X[num].sample(min(20000, len(X)), random_state=seed)
    corr = muestra.corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundantes = [c for c in superior.columns if (superior[c] > umbral_corr).any()]
    num = [c for c in num if c not in redundantes]
    informe["quitadas_por_correlacion"] = len(redundantes)
    informe["final"] = len(num) + len(cat)
    return num, cat, informe


def preprocesador_lineal(num, cat):
    """Para modelos sensibles a la escala (regresión logística, K-Means, PCA)."""
    numerico = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("esc", StandardScaler())])
    categorico = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=0.01, sparse_output=False)),
    ])
    return ColumnTransformer([("num", numerico, num), ("cat", categorico, cat)])


def preprocesador_arboles(num, cat):
    """Para modelos de árboles: no necesitan escalar y toleran valores nulos.

    Las categóricas salen como números enteros que el modelo trata como categorías.
    Las que aparecen en menos de 0,5 % se agrupan en una sola, así ninguna variable
    supera el máximo de categorías que admite el modelo.
    """
    categorico = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               min_frequency=0.005)),
    ])
    return ColumnTransformer([("num", "passthrough", num), ("cat", categorico, cat)])


def indices_categoricas(num, cat):
    """Posición de las categóricas en la salida de `preprocesador_arboles` (van después de las numéricas)."""
    return list(range(len(num), len(num) + len(cat)))
