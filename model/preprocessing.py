import os
import json
import joblib
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


"""Módulo de preprocesamiento.

Este módulo centraliza las transformaciones necesarias antes de entrenar el
modelo: carga del dataset limpio, construcción del `ColumnTransformer`,
división de datos, aplicación del ajuste/transformación y persistencia de
artefactos (preprocesador y nombres de características).

Notas:
- Los nombres de columnas y las listas `NUM_CONT`, `CAT_COLS`, `BIN_COLS` se
    usan para construir el pipeline y para reconstruir los nombres de
    características después de aplicar OneHotEncoding.
"""

# Columnas objetivo y por tipo
TARGET = "liver_cancer"
NUM_CONT = ["age", "bmi", "liver_function_score", "alpha_fetoprotein_level"]
BIN_COLS = ["hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "diabetes"]
CAT_COLS = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]


def load_cleaned_dataset(root_dir: str) -> pd.DataFrame:
    """Carga el CSV ya limpiado desde `data/processed/cleaned_dataset.csv`.

    Args:
        root_dir: ruta raíz del proyecto (normalmente el directorio padre de
            `model/`).

    Returns:
        DataFrame con el dataset limpio.

    Lanza FileNotFoundError si el archivo no existe.
    """
    path = os.path.join(root_dir, "data", "processed", "cleaned_dataset.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {path}")
    return pd.read_csv(path)


def make_preprocessor() -> ColumnTransformer:
    """Construye y devuelve un `ColumnTransformer` para el dataset.

    - Escala las columnas numéricas con `StandardScaler`.
    - Aplica `OneHotEncoder` a las columnas categóricas (ignorando valores
      desconocidos para permitir datos fuera de entrenamiento).
    - Pasa las columnas binarias tal cual (`passthrough`).

    Devuelve un transformer listo para `fit`/`transform`.
    """
    # OneHotEncoder: `sparse_output=False` para obtener arrays densos (np.ndarray)
    # compatibles con Keras/NumPy. `handle_unknown='ignore'` evita errores si en
    # el set de prueba aparece una categoría nueva.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
    scaler = StandardScaler()

    pre = ColumnTransformer(
        transformers=[
            ("num", scaler, NUM_CONT),
            ("cat", ohe, CAT_COLS),
            ("bin", "passthrough", BIN_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa el DataFrame en X/y y realiza un `train_test_split` estratificado.

    Args:
        df: DataFrame completo que incluye la columna objetivo `TARGET`.
        test_size: proporción para el conjunto de prueba.
        random_state: semilla para reproducibilidad.

    Returns:
        X_train, X_test, y_train, y_test
    """
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_transform_preprocessor(pre: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """Ajusta y transforma los conjuntos de entrenamiento y prueba.

    Además intenta reconstruir los nombres de las características resultantes
    (columnas numéricas, columnas categóricas expandidas por OHE y columnas
    binarias). Esto facilita el seguimiento de las features usadas por el
    modelo y su persistencia.

    Devuelve: X_train_transformed, X_test_transformed, feature_names (lista).
    """
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    # Reconstrucción de nombres de características: numéricas + cat-expanded + bin
    num_names = NUM_CONT
    cat_feature_names = []
    try:
        # Accedemos al OneHotEncoder dentro del ColumnTransformer
        ohe: OneHotEncoder = pre.named_transformers_["cat"]

        # sklearn API cambió entre versiones: get_feature_names_out o get_feature_names
        if hasattr(ohe, "get_feature_names_out"):
            cat_feature_names = ohe.get_feature_names_out(CAT_COLS).tolist()
        else:
            cat_feature_names = ohe.get_feature_names(CAT_COLS).tolist()
    except Exception:
        # En caso de error, dejamos la lista vacía para no interrumpir el flujo
        cat_feature_names = []
    feature_names = num_names + cat_feature_names + BIN_COLS
    return X_train_t, X_test_t, feature_names


def persist_artifacts(root_dir: str, version: str, pre: ColumnTransformer, feature_names: List[str]) -> str:
    """Persiste el preprocesador y los nombres de las features en disco.

    Se crea un subdirectorio en `saved_artifacts/<version>` donde se guarda:
    - `preprocessor.pkl` (joblib)
    - `feature_names.json` (lista de nombres de columnas final)

    Devuelve la ruta del directorio donde se guardaron los artefactos.
    """
    out_dir = os.path.join(root_dir, "saved_artifacts", version)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pre, os.path.join(out_dir, "preprocessor.pkl"))
    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, ensure_ascii=False, indent=2)
    return out_dir
