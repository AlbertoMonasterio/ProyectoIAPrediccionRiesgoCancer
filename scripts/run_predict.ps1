<#
Script: run_predict.ps1
Descripción: Arranca la API (uvicorn) usando el Python del venv detectado y abre la UI en el navegador.
Uso: Ejecutar desde la raíz del repo en PowerShell: .\scripts\run_predict.ps1
#>

Set-StrictMode -Version Latest

# Parámetro opcional: segundos máximos a esperar por la respuesta del servidor
param(
    [int]$WaitSeconds = 30
)

# Rutas posibles al ejecutable python del venv (añade más si tu equipo usa otra convención)
## Determine repository root (parent of the scripts folder)
$repoRoot = Split-Path -Parent $PSScriptRoot

# Rutas posibles al ejecutable python del venv (añade más si tu equipo usa otra convención)
$candidates = @(
    # relative to repository: .venv and venv
    [System.IO.Path]::Combine($repoRoot, '.venv', 'Scripts', 'python.exe'),
    [System.IO.Path]::Combine($repoRoot, 'venv', 'Scripts', 'python.exe'),
    # common absolute path used earlier
    'C:\venv\projenv\Scripts\python.exe'
)

function Find-PythonExecutable {
    foreach ($p in $candidates) {
        if (Test-Path $p) { return (Resolve-Path $p).Path }
    }
    return $null
}

$python = Find-PythonExecutable
if (-not $python) {
    Write-Host "No se encontró python del venv. Asegúrate de crear un venv y/o edita scripts/run_predict.ps1 para indicar su ruta." -ForegroundColor Red
    exit 1
}

Write-Host "Usando Python: $python"

# Iniciar uvicorn en un proceso separado
$args = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000')
Write-Host "Iniciando servidor uvicorn..."
$proc = Start-Process -FilePath $python -ArgumentList $args -PassThru

Write-Host "Esperando inicialización (esperando hasta $WaitSeconds segundos)..."
# Esperar activamente a que el puerto 8000 responda (máx $WaitSeconds s)
$maxWait = $WaitSeconds
$elapsed = 0
$tnc = $null
while ($elapsed -lt $maxWait) {
    try {
        $tnc = Test-NetConnection -ComputerName 127.0.0.1 -Port 8000 -WarningAction SilentlyContinue
    } catch {
        $tnc = $null
    }
    if ($tnc -and $tnc.TcpTestSucceeded) { break }
    Start-Sleep -Seconds 1
    $elapsed += 1
}

# Abrir navegador en la raíz (servido por FastAPI si frontend existe)
# Abrir la UI con el parámetro blank=1 para presentar el formulario vacío al usuario
$url = 'http://127.0.0.1:8000/?blank=1'
if ($tnc -and $tnc.TcpTestSucceeded) {
    Write-Host "Abriendo navegador en: $url"
    Start-Process $url
} else {
    Write-Host "Advertencia: el puerto 8000 no respondió tras $maxWait segundos. Abriendo navegador de todos modos: $url" -ForegroundColor Yellow
    Start-Process $url
}

Write-Host "Servidor iniciado (PID: $($proc.Id)). Presiona Enter para detenerlo..."
[void][System.Console]::ReadLine()

Write-Host "Deteniendo servidor..."
try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
Write-Host "Servidor detenido."
