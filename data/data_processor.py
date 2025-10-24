import os
import re
import sqlite3
from typing import List

import numpy as np
import pandas as pd

"""
Procesamiento de datos para el dataset de hígado (SQL) y guardado del scaler.

Objetivo de este archivo: SOLO limpieza de datos.

Notas importantes:
- El script SQL define la tabla "mytable" y usa age como PRIMARY KEY, pero hay edades repetidas.
	Para evitar errores de UNIQUE constraint, eliminamos el PRIMARY KEY al ejecutar el script en memoria.
- Hay valores no numéricos ('.') en columnas numéricas. Se convierten a NaN.
- NO se realizan splits, NO se escalan variables, y NO se hace One-Hot Encoding aquí.
"""


# --- Configuración y Constantes ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(ROOT_DIR, "synthetic_liver_cancer_dataset.sql")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")
TABLE_NAME = "mytable"
DROP_ROWS_WITH_ANY_NAN = True  # Si True, elimina filas que tengan cualquier NaN tras la coerción
DROP_EXACT_DUPLICATES = False  # Si True, elimina filas duplicadas exactas; si False, solo reporta

# Columnas continuas que escalaremos
NUMERIC_COLUMNS = [
	"age",
	"bmi",
	"liver_function_score",
	"alpha_fetoprotein_level",
	# binarios que deberían ser 0/1
	"hepatitis_b",
	"hepatitis_c",
	"cirrhosis_history",
	"family_history_cancer",
	"diabetes",
	"liver_cancer",
]

CATEGORICAL_COLUMNS = [
	"gender",
	"alcohol_consumption",
	"smoking_status",
	"physical_activity_level",
]


def _sanitize_sql(sql_script: str) -> str:
	"""Quita la restricción PRIMARY KEY para evitar errores por edades repetidas.

	- Reemplaza "PRIMARY KEY" por cadena vacía en el bloque CREATE TABLE.
	- Limpia posibles espacios múltiples.
	"""
	# Elimina PRIMARY KEY en cualquier lugar del script (bastante permisivo pero efectivo aquí)
	sanitized = re.sub(r"\bPRIMARY\s+KEY\b", "", sql_script, flags=re.IGNORECASE)
	# Elimina restricciones NOT NULL para permitir valores faltantes provenientes de '.'
	sanitized = re.sub(r"\bNOT\s+NULL\b", "", sanitized, flags=re.IGNORECASE)
	# Normaliza espacios múltiples en líneas del CREATE TABLE
	sanitized = re.sub(r"\s+\)\s*;", ")\n;", sanitized)
	# Reemplaza valores '.' solos en INSERTs por NULL para evitar errores de sintaxis
	sanitized = re.sub(r"(?<=,)\s*\.(?=\s*(,|\)))", " NULL", sanitized)
	return sanitized


def load_data_from_sql(db_script_path: str, table_name: str) -> pd.DataFrame:
	"""Carga los datos ejecutando el script SQL en una base en memoria.

	- Sanitiza el script para remover PRIMARY KEY en "age" (edades repetidas en el dataset).
	- Ejecuta CREATE TABLE + INSERTs.
	- Retorna un DataFrame con todos los registros de "table_name".
	"""
	print(f"-> Cargando datos desde: {db_script_path}...")
	if not os.path.isfile(db_script_path):
		raise FileNotFoundError(f"No se encontró el archivo SQL en {db_script_path}")

	conn = sqlite3.connect(":memory:")
	try:
		with open(db_script_path, "r", encoding="utf-8") as f:
			raw_sql = f.read()

		sql_script = _sanitize_sql(raw_sql)
		conn.executescript(sql_script)

		df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
		print(f"-> Datos cargados. Filas: {len(df)}, Columnas: {list(df.columns)}")
		return df
	except Exception as e:
		raise RuntimeError(f"ERROR al cargar los datos desde SQL: {e}")
	finally:
		conn.close()


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
	"""Convierte a numéricas las columnas especificadas, forzando errores a NaN."""
	for col in columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Aplica limpieza básica sin ingeniería de atributos.

	- Convierte columnas numéricas (incluyendo binarias) a tipo numérico (errores->NaN)
	- Normaliza columnas string (strip)
	- Opcionalmente elimina duplicados exactos (No)
	- Opcionalmente elimina filas con cualquier NaN (Si)
	- Devuelve el DataFrame limpio (con posibles NaN si había valores inválidos)
	"""
	if df is None or df.empty:
		raise ValueError("DataFrame vacío. No se puede limpiar.")

	print("-> Iniciando limpieza de datos...")

	# 1) Normalizar strings (strip)
	for col in df.columns:
		if col in CATEGORICAL_COLUMNS and col in df.columns:
			df[col] = df[col].astype(str).str.strip()

	# 2) Convertir numéricas (errores a NaN)
	df = _coerce_numeric(df, [c for c in NUMERIC_COLUMNS if c in df.columns])

	# 3) Duplicados exactos: opcional
	dup_count = df.duplicated().sum()
	if DROP_EXACT_DUPLICATES:
		before = len(df)
		df = df.drop_duplicates().reset_index(drop=True)
		removed = before - len(df)
		if removed:
			print(f"-> Duplicados eliminados: {removed}")
	else:
		if dup_count:
			print(f"-> Duplicados exactos detectados (no eliminados): {dup_count}")

	# 4) Reporte básico de NaNs
	nan_report = df.isna().sum()
	if nan_report.any():
		print("-> Valores faltantes por columna (post-coerción):")
		print(nan_report[nan_report > 0].sort_values(ascending=False))

	# 5) Eliminar filas con cualquier NaN si está habilitado
	if DROP_ROWS_WITH_ANY_NAN:
		before_rows = len(df)
		df = df.dropna().reset_index(drop=True)
		removed_rows = before_rows - len(df)
		print(f"-> Filas eliminadas por tener NaN: {removed_rows}")

	print("-> Limpieza completada.")
	return df


def save_cleaned_data(df: pd.DataFrame, path: str) -> None:
	"""Guarda el dataset limpio en CSV."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	df.to_csv(path, index=False)
	print(f"-> Dataset limpio guardado en: {path}")


def run_cleaning_pipeline():
	"""Carga desde SQL, aplica limpieza básica y guarda CSV limpio."""
	df = load_data_from_sql(DB_FILE, TABLE_NAME)
	df_clean = clean_data(df)

	print("\n--- Resumen post-limpieza ---")
	print("Primeras 5 filas:")
	print(df_clean.head())
	print("\nInfo rápida:")
	print(df_clean.dtypes)
	print(f"\nShape final (sin NaNs si DROP_ROWS_WITH_ANY_NAN=True): {df_clean.shape}")

	save_cleaned_data(df_clean, OUTPUT_CSV)
	return df_clean


if __name__ == "__main__":
	# Ejecutar SOLO limpieza al correr el archivo directamente
	run_cleaning_pipeline()

