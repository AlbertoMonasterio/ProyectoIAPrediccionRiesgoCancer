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

# Ruta raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_latest_artifact_dir(root: str):
    # Busca el directorio de artefactos más reciente
    pattern = os.path.join(root, "saved_artifacts", "model_v*")
    versions = glob.glob(pattern)
    if not versions:
        return None
    versions.sort(key=os.path.getmtime, reverse=True)
    return versions[0]


ARTIFACT_DIR = _get_latest_artifact_dir(ROOT)
if ARTIFACT_DIR is None:
    raise RuntimeError("No saved_artifacts found. Entrena el modelo primero para crear artefactos.")

# Cargar modelo y preprocesador entrenados
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.keras")
PREPROC_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")

model = keras.models.load_model(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)

# Cargar nombres de features (orden exacto que espera el modelo)
FEATURE_NAMES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.json")
FEATURE_NAMES = []
try:
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        _j = json.load(f)
        FEATURE_NAMES = _j.get("feature_names", [])
except Exception:
    FEATURE_NAMES = []


# Pydantic model que refleja las columnas de entrada
class PatientData(BaseModel):
    # Modelo de datos de entrada para la predicción
    age: float
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

# Permitir CORS para desarrollo (frontend local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático (index.html y archivos en /static)
FRONTEND_DIR = os.path.join(ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    # Endpoint raíz: sirve el archivo index.html del frontend
    from fastapi.responses import FileResponse
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/health")
def health():
    # Endpoint de salud: confirma que la API y los artefactos están disponibles
    return {"status": "ok", "artifact_dir": ARTIFACT_DIR}


@app.get("/meta")
def meta():
    # Devuelve metadatos del modelo para alinear el frontend
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
    # Endpoint principal de predicción
    # Recibe datos de paciente, normaliza y transforma los inputs, ejecuta el modelo y devuelve el riesgo estimado
    try:
        # Calcular BMI a partir de altura y peso
        height_m = data.height_cm / 100.0
        bmi = data.weight_kg / (height_m ** 2) if height_m > 0 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculando BMI: {e}")

    # Normalización de categorías: mapeo de etiquetas del formulario (ES) a categorías del modelo (EN)
    try:
        from sklearn.preprocessing import OneHotEncoder  # type: ignore
        ohe = preproc.named_transformers_.get("cat")
        cat_cols = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
        allowed = {}
        if hasattr(ohe, "categories_"):
            for col, cats in zip(cat_cols, ohe.categories_):
                allowed[col] = {str(c).strip().lower(): str(c) for c in cats}

        # Diccionarios de sinónimos para traducir etiquetas del frontend
        syn_gender = {"Masculino": "Male", "Femenino": "Female"}
        syn_alcohol = {"Nunca": "Never", "Ocasionalmente": "Occasional", "Regularmente": "Regular"}
        syn_smoke = {"Nunca": "Never", "Ex-fumador": "Former", "Actualmente": "Current"}
        syn_activity = {"Bajo (sedentario)": "Low", "Moderado": "Moderate", "Alto (activo)": "High"}

        def normalize_choice(value: str, mapping: dict, allowed_map: dict):
            # Normaliza una elección categórica usando sinónimos y mapeos permitidos
            if value is None:
                return value
            s = str(value).strip()
            if s in mapping:
                return mapping[s]
            low = s.lower()
            if low in allowed_map: 
                return allowed_map[low]
            for k, v in mapping.items():
                if k.lower() == low:
                    return v
            return s

        # Alias para usar abajo
        normalize = normalize_choice
        def normalize_alcohol(value: str, allowed_map: dict):
            return normalize_choice(value, syn_alcohol, allowed_map)

    except Exception:
        allowed = {}
        normalize = lambda v, syn, mapping: v  
        def normalize_alcohol(v, m):
            return v
        syn_gender = syn_alcohol = syn_smoke = syn_activity = {}

    # Construir el diccionario de entrada para el modelo
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

    # Transformar los datos usando el preprocesador
    try:
        X_t = preproc.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {e}")

    # Ejecutar la predicción con el modelo
    try:
        prob = float(model.predict(X_t).ravel()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    pct = round(prob * 100, 2)

    # Decisión de acción según el riesgo
    if pct <= 50:
        action = "Recomendación de seguimiento/chequeos."
    else:
        action = "Alerta: Cita clínica inmediata."

    result = {"risk_pct": pct, "action": action}
    if debug:
        result["normalized_input"] = row
        # Incluir categorías one-hot activas por columna categórica 
        try:
            from sklearn.preprocessing import OneHotEncoder  # type: ignore
            ohe = preproc.named_transformers_.get("cat")
            cat_cols = ["gender", "alcohol_consumption", "smoking_status", "physical_activity_level"]
            if hasattr(ohe, "get_feature_names_out"):
                cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
            else:
                cat_feature_names = list(ohe.get_feature_names(cat_cols))

            # Transformación solo del bloque categórico para alinear nombres y valores 1:1
            df_cat = df[cat_cols]
            cat_vec = ohe.transform(df_cat)
            # Convertir a lista plana para acceso cómodo
            if hasattr(cat_vec, "toarray"):
                cat_row = cat_vec.toarray()[0].tolist()
            else:
                cat_row = cat_vec[0].tolist()

            idx_map = {name: i for i, name in enumerate(cat_feature_names)}
            active = {}
            for col in cat_cols:
                prefix = f"{col}_"
                candidates = [n for n in cat_feature_names if n.startswith(prefix)]
                chosen = None
                for n in candidates:
                    i = idx_map[n]
                    if float(cat_row[i]) >= 0.5:
                        chosen = n[len(prefix):]
                        break
                active[col] = chosen
            result["encoded_onehot_active"] = active
        except Exception:
            # No bloquear si algo falla
            pass
    return result
