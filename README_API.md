# API - Cómo probar localmente 

Este archivo explica cómo  arrancar la API de predicción localmente en su máquina Windows (PowerShell) usando el modelo ya entrenado o entrenándolo ellos mismos.

## Requisitos
- Tener Python 3.10–3.13 instalado.
- Recomendado: ejecutar en una ruta corta para evitar problemas con rutas largas (OneDrive puede causar rutas largas). Usaremos un venv en `C:\venv\projenv` en los ejemplos.

## Pasos rápidos (recomendado)

1) Clonar el repo y situarte en la raíz del proyecto

2) Crear el entorno virtual en una ruta corta y activarlo

```powershell
python -m venv C:\venv\projenv
C:\venv\projenv\Scripts\Activate.ps1
```

Si PowerShell bloquea scripts, ejecuta (una sola vez):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3) Instalar dependencias

```powershell
C:\venv\projenv\Scripts\python -m pip install --upgrade pip setuptools wheel
C:\venv\projenv\Scripts\python -m pip install -r requirements.txt
```

4) Generar los artefactos del modelo (opcional)

Si quieres reproducir el entrenamiento y generar el modelo en `saved_artifacts/` (tarda dependiendo de tu máquina):

```powershell
C:\venv\projenv\Scripts\python -m model.train_mlp
```

Esto creará `saved_artifacts/model_vYYYYMMDD_HHMMSS` con:
- `model.keras` (modelo Keras)
- `preprocessor.pkl` (preprocesador)
- `feature_names.json`, `metrics.json`, `history.json`

5) Usar el modelo pre-entrenado (alternativa rápida)

Si no deseas entrenar, puedes usar la versión pre-entrenada localizada en `saved_artifacts/model_v20251025_122710` incluida en este repo (si está presente). El archivo `app/main.py` busca automáticamente la versión más reciente bajo `saved_artifacts/`.

6) Levantar la API (FastAPI / Uvicorn)

Opción recomendada (script que arranca el servidor y abre la UI vacía):

```powershell
# Desde la raíz del repo
.\scripts\run_predict.ps1

# Opciones:
#   -WaitSeconds <n>  -> controlar cuánto esperar a que el servidor responda antes de abrir el navegador (por defecto 30)
# Ejemplo: espera 5s y abre la UI vacía
.\scripts\run_predict.ps1 -WaitSeconds 5
```

Qué hace el script `run_predict.ps1`:

- Detecta el python del venv (revisa `.venv`, `venv` y `C:\venv\projenv`).
- Arranca uvicorn en background.
- Espera activamente hasta N segundos a que el puerto 8000 responda.
- Abre el navegador en `http://127.0.0.1:8000/?blank=1` (el `?blank=1` hace que el frontend presente el formulario vacío).
- Presionando Enter en la terminal el script detiene el servidor.

Si prefieres arrancar manualmente (ver logs en la terminal):

```powershell
C:\venv\projenv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Visita `http://127.0.0.1:8000/docs` para probar la API desde Swagger UI o abre la UI en `/` para usar el formulario.

7) Probar el endpoint `/predict`

Usa el script de ejemplo (con el venv activado):

```powershell
python client_examples\predict_example.py
```

O con PowerShell directamente usando `Invoke-RestMethod` (ejemplo rápido). IMPORTANTE: el backend actual espera `height_cm` y `weight_kg` en vez de `bmi`.

```powershell
$body = @{
  age = 55; height_cm = 170; weight_kg = 80; liver_function_score = 1.2; alpha_fetoprotein_level = 3.4;
  hepatitis_b = 0; hepatitis_c = 0; cirrhosis_history = 0; family_history_cancer = 0; diabetes = 0;
  gender = 'male'; alcohol_consumption = 'low'; smoking_status = 'never'; physical_activity_level = 'moderate'
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType 'application/json'
```

## Notas importantes
- `app/main.py` intenta cargar la versión más reciente en `saved_artifacts/`. Si el directorio no existe, primero entrena el modelo con `train_mlp.py`.
- No recomendamos subir modelos pesados al repositorio; para compartir modelos entre compañeros puedes:
  - Adjuntar el zip de `saved_artifacts/model_vYYYY...` en la release del repo, o
  - Subir los artefactos a un almacenamiento compartido (Drive, S3) y añadir un script `scripts/download_artifacts.ps1` para descargarlos.

## Consejos para VS Code
- Selecciona el intérprete Python del proyecto: Ctrl+Shift+P → "Python: Select Interpreter" → apunta a `C:\venv\projenv\Scripts\python.exe`.
- Asegúrate de que la terminal de VS Code active el entorno automáticamente en nuevas terminales (setting `python.terminal.activateEnvironment: true`).


