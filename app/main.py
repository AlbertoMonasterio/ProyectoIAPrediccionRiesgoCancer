from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import unicodedata
import glob
import joblib
import json
import pandas as pd
from tensorflow import keras

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_latest_artifact_dir(root: str):
    pattern = os.path.join(root, "saved_artifacts", "model_v*")
    versions = glob.glob(pattern)
    if not versions:
        return None
    versions.sort(key=os.path.getmtime, reverse=True)
    return versions[0]


ARTIFACT_DIR = _get_latest_artifact_dir(ROOT)
if ARTIFACT_DIR is None:
    raise RuntimeError("No saved_artifacts found. Entrena el modelo primero para crear artefactos.")

# Cargar artefactos al inicio
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.keras")
PREPROC_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

model = keras.models.load_model(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)


# Pydantic model que refleja las columnas de entrada
class PatientData(BaseModel):
    age: float
    # accept height (cm) and weight (kg) instead of BMI for user-friendly input
    height_cm: float
    weight_kg: float
    liver_function_score: float
    alpha_fetoprotein_level: float
    hepatitis_b: int
    hepatitis_c: int
    cirrhosis_history: int
    family_history_cancer: int
    diabetes: int
    gender: str
    alcohol_consumption: str
    smoking_status: str
    physical_activity_level: str


app = FastAPI(title="Predicción Riesgo Cáncer Hígado - API")

# Allow CORS for frontend apps during development. For production, lock this down to the
# specific origins that will host the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático (index.html) desde / para facilitar pruebas locales
FRONTEND_DIR = os.path.join(ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    # mount static files under /static and serve index.html at /
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    # return the frontend index.html
    from fastapi.responses import FileResponse
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"status": "ok", "artifact_dir": ARTIFACT_DIR}


@app.get("/meta")
def meta():
    """Devuelve metadatos del modelo para alinear el frontend: columnas categóricas y sus categorías,
    además de columnas numéricas y binarias (si aplica).
    """
    cat_cols = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
    categories = {}
    try:
        ohe = preproc.named_transformers_.get("cat")
        if hasattr(ohe, "categories_"):
            for col, cats in zip(cat_cols, ohe.categories_):
                categories[col] = list(map(str, cats))
    except Exception:
        categories = {}
    return {
        "categories": categories,
        "notes": "Las categorías devueltas reflejan exactamente lo conocido por el modelo."
    }


@app.post("/predict")
def predict(data: PatientData, debug: bool = False):
    # Convertir los inputs: calcular BMI a partir de altura y peso y crear el dict esperado
    try:
        height_m = data.height_cm / 100.0
        bmi = data.weight_kg / (height_m ** 2) if height_m > 0 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculando BMI: {e}")

    # Normalización de categorías: mapeo directo de etiquetas fijas del formulario (ES) a categorías del modelo (EN)
    try:
        from sklearn.preprocessing import OneHotEncoder  # type: ignore
        ohe = preproc.named_transformers_.get("cat")
        cat_cols = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
        allowed = {}
        if hasattr(ohe, "categories_"):
            for col, cats in zip(cat_cols, ohe.categories_):
                allowed[col] = {str(c).strip().lower(): str(c) for c in cats}

        # Etiquetas fijas que envía el formulario (ES) -> categorías del modelo (EN)
        syn_gender = {"Masculino": "Male", "Femenino": "Female"}
        syn_alcohol = {"Nunca": "Never", "Ocasionalmente": "Occasional", "Regularmente": "Regular"}
        syn_smoke = {"Nunca": "Never", "Ex-fumador": "Former", "Actualmente": "Current"}
        syn_activity = {"Bajo (sedentario)": "Low", "Moderado": "Moderate", "Alto (activo)": "High"}

        def normalize_choice(value: str, mapping: dict, allowed_map: dict):
            if value is None:
                return value
            s = str(value).strip()
            if s in mapping:
                return mapping[s]
            low = s.lower()
            if low in allowed_map:  # ya viene en EN (cualquier casing)
                return allowed_map[low]
            # fallback case-insensitive contra mapping
            for k, v in mapping.items():
                if k.lower() == low:
                    return v
            return s

        # alias para usar abajo
        normalize = normalize_choice
        def normalize_alcohol(value: str, allowed_map: dict):
            return normalize_choice(value, syn_alcohol, allowed_map)

    except Exception:
        # si algo falla, seguimos sin normalizar (preproc.handle_unknown='ignore' mitigará)
        allowed = {}
        normalize = lambda v, syn, mapping: v  # type: ignore
        def normalize_alcohol(v, m):
            return v
        syn_gender = syn_alcohol = syn_smoke = syn_activity = {}

    row = {
        "age": data.age,
        "bmi": bmi,
        "liver_function_score": data.liver_function_score,
        "alpha_fetoprotein_level": data.alpha_fetoprotein_level,
        "hepatitis_b": data.hepatitis_b,
        "hepatitis_c": data.hepatitis_c,
        "cirrhosis_history": data.cirrhosis_history,
        "family_history_cancer": data.family_history_cancer,
        "diabetes": data.diabetes,
        "gender": normalize(getattr(data, "gender", None), syn_gender, allowed.get("gender", {})),
        "alcohol_consumption": normalize_alcohol(getattr(data, "alcohol_consumption", None), allowed.get("alcohol_consumption", {})),
        "smoking_status": normalize(getattr(data, "smoking_status", None), syn_smoke, allowed.get("smoking_status", {})),
        "physical_activity_level": normalize(getattr(data, "physical_activity_level", None), syn_activity, allowed.get("physical_activity_level", {})),
    }

    df = pd.DataFrame([row])

    try:
        X_t = preproc.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {e}")

    try:
        prob = float(model.predict(X_t).ravel()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    pct = round(prob * 100, 2)

    if pct <= 50:
        action = "Recomendación de seguimiento/chequeos."
    else:
        action = "Alerta: Cita clínica inmediata."

    result = {"risk_pct": pct, "action": action}
    if debug:
        result["normalized_input"] = row
    return result
