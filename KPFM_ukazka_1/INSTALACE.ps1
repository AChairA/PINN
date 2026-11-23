param(
    [string]$Python = "python",
    [switch]$VcetnePdf
)
$ErrorActionPreference = "Stop"
$environmentPath = Join-Path $PSScriptRoot ".venv"
& $Python -m venv $environmentPath
if ($LASTEXITCODE -ne 0) { throw "Nepodarilo se vytvorit prostredi Pythonu." }
$localPython = Join-Path $environmentPath "Scripts\python.exe"
& $localPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Instalace numerickych knihoven selhala." }
if ($VcetnePdf) {
    & $localPython -m pip install -r (Join-Path $PSScriptRoot "dokumentace\requirements_pdf.txt")
    if ($LASTEXITCODE -ne 0) { throw "Instalace knihoven pro PDF selhala." }
}
Write-Host "Pripraveno. Spustte .\SPUSTIT.ps1 -Rezim kontrola nebo -Rezim vse."
