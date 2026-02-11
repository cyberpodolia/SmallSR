param(
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null

if (-not (Test-Path ".venv")) {
    if (Get-Command $PythonLauncher -ErrorAction SilentlyContinue) {
        & $PythonLauncher -3.11 -m venv .venv
    } else {
        python -m venv .venv
    }
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r python/requirements.base.txt
python -m pip install -r python/requirements.win.txt

Write-Host "Venv ready. To run cam_server: scripts\\run_cam_server.ps1"
