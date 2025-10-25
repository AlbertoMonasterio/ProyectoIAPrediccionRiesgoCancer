import os, json, joblib
import pandas as pd
from tensorflow import keras

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
versions = sorted([os.path.join(ROOT, "saved_artifacts", d) for d in os.listdir(os.path.join(ROOT, "saved_artifacts")) if d.startswith("model_v")], key=os.path.getmtime, reverse=True)
ARTIFACT_DIR = versions[0]

model = keras.models.load_model(os.path.join(ARTIFACT_DIR, "model.keras"))
preproc = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))

payload = {
    "age": 55,
    "bmi": 27.5,
    "liver_function_score": 1.2,
    "alpha_fetoprotein_level": 3.4,
    "hepatitis_b": 0,
    "hepatitis_c": 0,
    "cirrhosis_history": 0,
    "family_history_cancer": 0,
    "diabetes": 0,
    "gender": "male",
    "alcohol_consumption": "low",
    "smoking_status": "never",
    "physical_activity_level": "moderate",
}

df = pd.DataFrame([payload])
X_t = preproc.transform(df)
prob = float(model.predict(X_t).ravel()[0])
pct = round(prob * 100, 2)
action = "Recomendación de seguimiento/chequeos." if pct <= 50 else "Alerta: Cita clínica inmediata."

result = {"risk_pct": pct, "action": action}
print(json.dumps(result))
