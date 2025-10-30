import os
import json
import joblib
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Columns
TARGET = "liver_cancer"
NUM_CONT = ["age", "bmi", "liver_function_score", "alpha_fetoprotein_level"]
BIN_COLS = ["hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "diabetes"]
CAT_COLS = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]


def load_cleaned_dataset(root_dir: str) -> pd.DataFrame:
    path = os.path.join(root_dir, "data", "processed", "cleaned_dataset.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {path}")
    return pd.read_csv(path)


def make_preprocessor() -> ColumnTransformer:
   
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
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_transform_preprocessor(pre: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    # Feature names (num + cat-expanded + bin)
    num_names = NUM_CONT
    cat_feature_names = []
    try:
        ohe: OneHotEncoder = pre.named_transformers_["cat"]
        
        if hasattr(ohe, "get_feature_names_out"):
            cat_feature_names = ohe.get_feature_names_out(CAT_COLS).tolist()
        else:
            cat_feature_names = ohe.get_feature_names(CAT_COLS).tolist()
    except Exception:
        cat_feature_names = []
    feature_names = num_names + cat_feature_names + BIN_COLS
    return X_train_t, X_test_t, feature_names


def persist_artifacts(root_dir: str, version: str, pre: ColumnTransformer, feature_names: List[str]) -> str:
    out_dir = os.path.join(root_dir, "saved_artifacts", version)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pre, os.path.join(out_dir, "preprocessor.pkl"))
    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, ensure_ascii=False, indent=2)
    return out_dir
