param(
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null

& .\.venv\Scripts\Activate.ps1
python python/cam_server/run_cam_server.py --port $Port
