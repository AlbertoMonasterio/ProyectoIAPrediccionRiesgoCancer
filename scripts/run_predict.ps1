<#
Script: run_predict.ps1
Descripción: Arranca la API (uvicorn) usando el Python del venv detectado y abre la UI en el navegador.
Uso: Ejecutar desde la raíz del repo en PowerShell: .\scripts\run_predict.ps1
#>

[CmdletBinding()]
param(
    # Segundos máximos a esperar por la respuesta del servidor
    [int]$WaitSeconds = 30,
    # Ruta opcional al ejecutable python del venv
    [string]$PythonPath
)

Set-StrictMode -Version Latest


## Determine repository root (parent of the scripts folder)
$repoRoot = Split-Path -Parent $PSScriptRoot

# Rutas posibles al ejecutable python del venv 
$candidates = @(
    # Preferir primero el venv corto del usuario
    'C:\Users\sayag\venvs\iaproj\Scripts\python.exe',
    'C:\venv\projenv\Scripts\python.exe',
    # y luego venvs dentro del repo 
    [System.IO.Path]::Combine($repoRoot, '.venv', 'Scripts', 'python.exe'),
    [System.IO.Path]::Combine($repoRoot, 'venv', 'Scripts', 'python.exe')
)

function Find-PythonExecutable {
    foreach ($p in $candidates) {
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    return $null
}

if ($PythonPath) {
    if (Test-Path $PythonPath) {
        $python = (Resolve-Path $PythonPath).Path
    } else {
        Write-Host "La ruta especificada en -PythonPath no existe: $PythonPath" -ForegroundColor Red
        exit 1
    }
} else {
    $python = Find-PythonExecutable
}
if (-not $python) {
    Write-Host "No se encontró python del venv. Asegúrate de crear un venv y/o edita scripts/run_predict.ps1 para indicar su ruta." -ForegroundColor Red
    exit 1
}

Write-Host "Usando Python: $python"

# Verificar artefactos antes de lanzar el servidor (fail-fast)
$artifactsRoot = Join-Path $repoRoot 'saved_artifacts'
if (-not (Test-Path $artifactsRoot)) {
    Write-Host "No existe la carpeta de artefactos: $artifactsRoot. Entrena primero el modelo (model/train_mlp.py)." -ForegroundColor Red
    exit 1
}

$latest = Get-ChildItem -Path $artifactsRoot -Filter 'model_v*' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) {
    Write-Host "No se encontraron versiones de artefactos en $artifactsRoot. Entrena primero el modelo." -ForegroundColor Red
    exit 1
}

$modelPath = Join-Path $latest.FullName 'model.keras'
$preprocPath = Join-Path $latest.FullName 'preprocessor.pkl'
if (-not (Test-Path $modelPath) -or -not (Test-Path $preprocPath)) {
    Write-Host "Faltan archivos requeridos en $($latest.FullName). Se esperan model.keras y preprocessor.pkl." -ForegroundColor Red
    exit 1
}

# Iniciar uvicorn en un proceso separado
$args = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000')

# Mensaje amigable para el usuario sobre el arranque y los warnings
Write-Host "Iniciando servidor uvicorn..."
Write-Host "Esperando a que el modelo y la API se inicialicen. Esto puede tardar varios segundos si el modelo es grande."
Write-Host "Si ves varios 'WARNING: TCP connect...', espera hasta que se abra el navegador o aparezca 'Servidor iniciado'.\n"

# Capturar logs para diagnóstico si el proceso termina prematuramente
$logsDir = Join-Path $repoRoot 'logs'
New-Item -ItemType Directory -Path $logsDir -ErrorAction SilentlyContinue | Out-Null
$outLog = Join-Path $logsDir 'uvicorn_stdout.log'
$errLog = Join-Path $logsDir 'uvicorn_stderr.log'

$proc = Start-Process -FilePath $python -ArgumentList $args -PassThru -WorkingDirectory $repoRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog

Write-Host "Esperando inicialización (esperando hasta $WaitSeconds segundos)..."
# Esperar activamente a que el puerto 8000 responda (máx $WaitSeconds s) y abortar si el proceso muere
$maxWait = $WaitSeconds
$elapsed = 0
$portReady = $false
while ($elapsed -lt $maxWait) {
    if ($proc.HasExited) {
        Write-Host "Uvicorn terminó antes de estar listo. Revisa artefactos, dependencias o errores en app/main.py." -ForegroundColor Red
        Write-Host "Últimas líneas de logs (stdout):" -ForegroundColor Yellow
        if (Test-Path $outLog) { Get-Content $outLog -Tail 40 }
        Write-Host "Últimas líneas de logs (stderr):" -ForegroundColor Yellow
        if (Test-Path $errLog) { Get-Content $errLog -Tail 40 }
        exit 1
    }
    $ok = $false
    try {
        $ok = Test-NetConnection -ComputerName 127.0.0.1 -Port 8000 -InformationLevel Quiet
    } catch {
        $ok = $false
    }
    if ($ok) { $portReady = $true; break }
    Start-Sleep -Seconds 1
    $elapsed += 1
}

# Abrir navegador en la raíz (servido por FastAPI si frontend existe)
# Abrir la UI con el parámetro blank=1 para presentar el formulario vacío al usuario
$url = 'http://127.0.0.1:8000/?blank=1'
if ($portReady) {
    Write-Host "Abriendo navegador en: $url"
    Start-Process $url
} else {
    Write-Host "Advertencia: el puerto 8000 no respondió tras $maxWait segundos. No se abrirá el navegador. Revisa logs de Uvicorn." -ForegroundColor Yellow
}

Write-Host "Servidor iniciado (PID: $($proc.Id)). Presiona Enter para detenerlo..."
[void][System.Console]::ReadLine()

Write-Host "Deteniendo servidor..."
try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
Write-Host "Servidor detenido."
