import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

import keras
from keras import layers
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

from .preprocessing import load_cleaned_dataset, make_preprocessor, split_data, fit_transform_preprocessor, persist_artifacts, TARGET


"""Entrenamiento de un MLP para la predicción de riesgo de cáncer de hígado.

Este módulo contiene funciones para construir un MLP simple con Keras,
entrenarlo usando los datos preprocesados y guardar los artefactos
resultantes (modelo, métricas e historial).

Flujo principal en `train_and_evaluate`:
1. Cargar dataset limpio.
2. Dividir datos (estratificado).
3. Construir y ajustar el preprocesador.
4. Construir el modelo, entrenar con callbacks y validar.
5. Evaluar en el conjunto de prueba y guardar métricas/historia.
"""


def build_mlp(input_dim: int) -> keras.Model:
    """Construye y compila un MLP sencillo.

    Arquitectura:
    - Input -> Dense(64, relu) -> Dropout(0.2)
    - Dense(32, relu) -> Dropout(0.2)
    - Dense(1, sigmoid)

    El modelo se compila con Adam, pérdida de entropía binaria y métricas de
    exactitud y AUC (que usamos para elegir el mejor checkpoint).

    Args:
        input_dim: número de features de entrada.

    Returns:
        Modelo compilado listo para `.fit()`.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_and_evaluate(root_dir: str) -> None:
    """Ejecuta el flujo completo de entrenamiento y evaluación.

    Pasos:
    - Cargar dataset limpio.
    - Dividir en train/test.
    - Construir y ajustar el preprocesador (se guardará como artefacto).
    - Construir el modelo, entrenar con validación y callbacks.
    - Evaluar en test, calcular métricas adicionales y persistir métricas e
      historial.
    """
    df = load_cleaned_dataset(root_dir)

    # Splits: train/test estratificado por la variable objetivo
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocess: construir preprocesador y transformar datos
    pre = make_preprocessor()
    X_train_t, X_test_t, feature_names = fit_transform_preprocessor(pre, X_train, X_test)

    # Build model: el input_dim viene del número de columnas resultantes
    model = build_mlp(X_train_t.shape[1])

    # Artifacts dir: usamos un timestamp para versionar los resultados
    version = datetime.now().strftime("model_v%Y%m%d_%H%M%S")
    out_dir = persist_artifacts(root_dir, version, pre, feature_names)

    # Callbacks: EarlyStopping + checkpoint basado en AUC de validación
    ckpt_path = os.path.join(out_dir, "model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True),
    ]

    # Train: usamos validation_split para mantener simple el flujo de ejemplo
    history = model.fit(
        X_train_t, y_train.values,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate: métricas desde Keras y cálculos adicionales con sklearn
    test_metrics = model.evaluate(X_test_t, y_test.values, verbose=0)
    y_prob = model.predict(X_test_t, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Extra metrics (más detalladas): ROC-AUC, PR-AUC, matriz de confusión, reporte
    roc_auc = roc_auc_score(y_test.values, y_prob)
    pr_auc = average_precision_score(y_test.values, y_prob)
    cm = confusion_matrix(y_test.values, y_pred)
    cls_rep = classification_report(y_test.values, y_pred, output_dict=True)

    # Save metrics & history
    metrics = {
        "keras_evaluate": {"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1]), "auc": float(test_metrics[2])},
        "sklearn": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "confusion_matrix": cm.tolist(), "classification_report": cls_rep},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save training history (convertir a tipos nativos para JSON)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(ROOT_DIR)  # go up to project root
    train_and_evaluate(ROOT_DIR)
