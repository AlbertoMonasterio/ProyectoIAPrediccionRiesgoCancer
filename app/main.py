from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
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


@app.post("/predict")
def predict(data: PatientData):
    # Convertir los inputs: calcular BMI a partir de altura y peso y crear el dict esperado
    try:
        height_m = data.height_cm / 100.0
        bmi = data.weight_kg / (height_m ** 2) if height_m > 0 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculando BMI: {e}")

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
        "gender": data.gender,
        "alcohol_consumption": data.alcohol_consumption,
        "smoking_status": data.smoking_status,
        "physical_activity_level": data.physical_activity_level,
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

    return {"risk_pct": pct, "action": action}
