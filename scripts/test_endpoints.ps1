param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$DebugPayload
)

function Write-Section($title) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
}

function Invoke-Endpoint {
    param(
        [Parameter(Mandatory=$true)][ValidateSet('GET','POST')][string]$Method,
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter()][object]$Body
    )
    try {
        if ($Method -eq 'GET') {
            return Invoke-RestMethod -Uri $Url -Method GET -TimeoutSec 20
        } else {
            if ($null -ne $Body) {
                $json = $Body | ConvertTo-Json -Depth 5
                return Invoke-RestMethod -Uri $Url -Method POST -Body $json -ContentType 'application/json' -TimeoutSec 30
            } else {
                return Invoke-RestMethod -Uri $Url -Method POST -TimeoutSec 30
            }
        }
    }
    catch {
        Write-Host "Error calling $Url" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor DarkRed
        if ($_.Exception.Response -and $_.Exception.Response.ContentLength -gt 0) {
            try {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $respBody = $reader.ReadToEnd()
                Write-Host "Response body:" -ForegroundColor DarkYellow
                Write-Host $respBody
            } catch {}
        }
        return $null
    }
}

# 1) /health
Write-Section "/health"
$health = Invoke-Endpoint -Method GET -Url "$BaseUrl/health"
if ($health) { $health | ConvertTo-Json -Depth 5 | Write-Host }

# 2) /meta
Write-Section "/meta"
$meta = Invoke-Endpoint -Method GET -Url "$BaseUrl/meta"
if ($meta) { $meta | ConvertTo-Json -Depth 5 | Write-Host }

# 3) /predict
Write-Section "/predict"
$payload = @{
  age = 55
  height_cm = 170
  weight_kg = 80
  liver_function_score = 1.2
  alpha_fetoprotein_level = 3.4
  hepatitis_b = 0
  hepatitis_c = 0
  cirrhosis_history = 0
  family_history_cancer = 0
  diabetes = 0
  gender = 'Masculino'                 # mapea a 'Male'
  alcohol_consumption = 'Nunca'         # mapea a 'Never'
  smoking_status = 'Nunca'              # mapea a 'Never'
  physical_activity_level = 'Moderado'  # mapea a 'Moderate'
}

# Mostrar payload que se enviará
Write-Section "Payload enviado (formulario)"
$payload | ConvertTo-Json -Depth 5 | Write-Host

if ($DebugPayload) {
    # añade ?debug=true para ver input normalizado
    $pred = Invoke-Endpoint -Method POST -Url "$BaseUrl/predict?debug=true" -Body $payload
} else {
    $pred = Invoke-Endpoint -Method POST -Url "$BaseUrl/predict" -Body $payload
}

if ($pred) {
    # Resumen de predicción
    Write-Section "Resultado"
    if ($pred.risk_pct -ne $null -and $pred.action -ne $null) {
        Write-Host ("Riesgo: {0}%" -f [math]::Round([double]$pred.risk_pct,2))
        Write-Host ("Acción: {0}" -f $pred.action)
    }

    # Entrada normalizada devuelta por el backend (solo si -DebugPayload)
    if ($pred.normalized_input) {
        Write-Section "Entrada normalizada (backend)"
        $pred.normalized_input | ConvertTo-Json -Depth 5 | Write-Host
    }

    # Cuerpo completo por si hace falta
    Write-Section "Respuesta completa"
    $pred | ConvertTo-Json -Depth 5 | Write-Host
}

Write-Host "`nListo. Si algún endpoint falló, asegúrate de que el servidor esté corriendo (scripts\\run_predict.ps1) y revisa logs/uvicorn_*.log." -ForegroundColor Green
