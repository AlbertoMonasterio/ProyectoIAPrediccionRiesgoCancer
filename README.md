# API — Cómo ejecutar y probar localmente (Windows / PowerShell)

Esta guía muestra pasos claros y comprobables para poner en marcha la API de predicción localmente en Windows (PowerShell). Incluye opciones rápidas usando un modelo preentrenado y pasos para reproducir el entrenamiento si necesitas generar los artefactos.
## Resumen rápido

- Requisitos: Python 3.10–3.13.
- Recomendación: usar un entorno virtual en una ruta corta (por ejemplo `C:\venv\projenv`) para evitar problemas con rutas largas.
- Pasos principales: crear/activar venv → instalar dependencias → asegurarse de tener artefactos en `saved_artifacts/` (o entrenar) → arrancar la API → probar `/predict`.

---

## 1) Requisitos

- Python 3.10–3.13.
- Git (opcional).
- PowerShell (v5.1 o superior). Si PowerShell bloquea la ejecución de scripts, necesitarás cambiar la política de ejecución (ver sección más abajo).

Archivos y carpetas clave del repo:

- `app/` — FastAPI app (arranque y endpoints).
- `model/` — código de preprocesamiento y entrenamiento (`train_mlp.py`).
- `saved_artifacts/` — artefactos del modelo (por ejemplo `model.keras`, `preprocessor.pkl`, `feature_names.json`).
- `client_examples/` — scripts de ejemplo para invocar la API.
- `scripts/run_predict.ps1` — script recomendado para arrancar el servidor y abrir el navegador.

> Nota importante: el servidor espera habitualmente que bajo `saved_artifacts/model_v*/` existan al menos `model.keras` y `preprocessor.pkl`. Si falta `preprocessor.pkl`, deberás generar los artefactos entrenando el modelo (ver sección 4).

---

## 2) Inicio rápido — comandos PowerShell

1) Crear y activar el entorno virtual (ruta de ejemplo):

```powershell
python -m venv C:\venv\projenv
C:\venv\projenv\Scripts\Activate.ps1
```

Si PowerShell impide ejecutar scripts, ejecutar (una vez):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2) Actualizar pip e instalar dependencias:

```powershell
C:\venv\projenv\Scripts\python -m pip install --upgrade pip setuptools wheel
C:\venv\projenv\Scripts\python -m pip install -r requirements.txt
```

3) Ejecutar la API (opción recomendada — script que gestiona checks y abre UI):

```powershell
# Desde la raíz del repositorio
.\scripts\run_predict.ps1

# Opcional: esperar menos/mas antes de abrir navegador, por ejemplo 10 segundos
.\scripts\run_predict.ps1 -WaitSeconds 10
```

4) Alternativa: arrancar Uvicorn manualmente (útil para ver logs directo en la terminal):


```powershell
C:\venv\projenv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Nota sobre los warnings y el tiempo de arranque:**

Es normal que al iniciar el servidor (especialmente usando el script PowerShell) aparezcan mensajes como:

```
WARNING: TCP connect to (127.0.0.1 : 8000) failed
```

Esto ocurre porque el script verifica repetidamente si la API está lista antes de abrir el navegador. Si el modelo es grande, la carga puede tardar varios segundos y los warnings se mostrarán hasta que el servidor esté disponible. No indican un error, solo que el backend aún está arrancando.

**Recomendación:**
- Espera hasta que se abra el navegador o veas el mensaje "Servidor iniciado".
- Si el modelo es muy grande, el arranque puede tardar más de 10 segundos.
- Si tienes dudas, revisa los logs en la carpeta `logs/`.

Visita `http://127.0.0.1:8000/docs` para la documentación interactiva (Swagger UI) o abre `/` para la UI incluida.

---

## 3) Probar el endpoint `/predict`

- Con el entorno activado, usar el script de ejemplo:

```powershell
python client_examples\predict_example.py
```

- Ejemplo rápido con `Invoke-RestMethod` (PowerShell): el backend actualmente espera campos como `height_cm` y `weight_kg` (no `bmi`):

```powershell
$body = @{
  age = 55; height_cm = 170; weight_kg = 80; liver_function_score = 1.2; alpha_fetoprotein_level = 3.4;
  hepatitis_b = 0; hepatitis_c = 0; cirrhosis_history = 0; family_history_cancer = 0; diabetes = 0;
  gender = 'male'; alcohol_consumption = 'low'; smoking_status = 'never'; physical_activity_level = 'moderate'
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType 'application/json'
```

---

## 4) Artefactos del modelo — usar preentrenado o entrenar localmente

Opción A — Usar artefactos ya generados

- Si el repo incluye una carpeta bajo `saved_artifacts/` con `model.keras` y `preprocessor.pkl`, `app/main.py` intentará cargar la versión más reciente automáticamente.

Opción B — Reentrenar y generar artefactos (recomendado si falta `preprocessor.pkl` o quieres reproducir resultados)

```powershell
# Con el venv activado
C:\venv\projenv\Scripts\python -m model.train_mlp
```

Esto debería crear un nuevo subdirectorio en `saved_artifacts/` con:

- `model.keras` — el modelo Keras entrenado.
- `preprocessor.pkl` — objeto de preprocesado necesario para las predicciones.
- `feature_names.json`, `metrics.json`, `history.json`.

Si tras esto faltara algún archivo necesario, revisa la salida del entrenamiento para errores y abre `model/train_mlp.py` para ver qué artefactos guarda.

---

## 5) Diagnóstico y resolución de problemas comunes

- Error: servidor no arranca o `run_predict.ps1` aborta por falta de artefactos.
  - Revisa `saved_artifacts/` y comprueba que exista al menos una carpeta `model_v*` con `model.keras` y `preprocessor.pkl`.
  - Si falta `preprocessor.pkl`, ejecuta el entrenamiento (`python -m model.train_mlp`).

- Error: puerto 8000 ocupado o Uvicorn falla.
  - Verifica procesos usando el puerto y mata procesos que estén usando 8000.
  - Lanza Uvicorn manualmente para ver errores en la terminal.

- Error: dependencias faltantes o versiones incompatibles.
  - Asegúrate de instalar `requirements.txt` con el Python del venv corto.

- Logs:
  - `scripts/run_predict.ps1` almacena salida en `logs/uvicorn_stdout.log` y `logs/uvicorn_stderr.log` (si el script se usa). Abre esos archivos para diagnósticos.

---

## 6) Checklist mínima para dejar el proyecto en funcionamiento

1) Tener Python 3.10–3.13 instalado y accesible.
2) Crear y activar un venv en una ruta corta (ej. `C:\venv\projenv`).
3) Instalar dependencias: `pip install -r requirements.txt` usando el Python del venv.
4) Verificar `saved_artifacts/`:
   - Si existe una versión con `model.keras` y `preprocessor.pkl`, OK.
   - Si falta `preprocessor.pkl` o no hay artefactos: ejecutar `python -m model.train_mlp` para generarlos.
5) Arrancar la API con `scripts/run_predict.ps1` o `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`.
6) Probar `/predict` con `client_examples/predict_example.py` o `Invoke-RestMethod`.

---

## 7) Consejos para VS Code

- Selecciona el intérprete (Ctrl+Shift+P → "Python: Select Interpreter") apuntando al Python dentro del venv (`C:\venv\projenv\Scripts\python.exe`).
- Habilita la activación automática del entorno en la terminal: `python.terminal.activateEnvironment: true`.

---


---

Pulsa Enter para detener `run_predict.ps1` si lo usas para arrancar el servidor.

