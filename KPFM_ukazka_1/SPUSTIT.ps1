param(
    [ValidateSet("kontrola", "vse", "trenink", "citlivosti", "testy", "pdf")]
    [string]$Rezim = "kontrola",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
if (-not $Python) {
    $localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { "python" }
}
$casePath = Join-Path $PSScriptRoot "ukazka_kpfm_D1e-15"
switch ($Rezim) {
    "kontrola" { & $Python (Join-Path $PSScriptRoot "dokumentace\over_balicek.py") }
    "vse" { & $Python (Join-Path $casePath "reproduce.py") }
    "trenink" { & $Python (Join-Path $casePath "run_demo.py") fit }
    "citlivosti" { & $Python (Join-Path $casePath "time_sensitivity.py") }
    "testy" { & $Python -m unittest discover -s $casePath -p test_demo.py -v }
    "pdf" { & $Python (Join-Path $PSScriptRoot "dokumentace\vytvor_pdf.py") }
}
if ($LASTEXITCODE -ne 0) { throw "Beh skoncil s chybou ($LASTEXITCODE)." }
