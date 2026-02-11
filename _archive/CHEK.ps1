<#  check_onnx_cuda_env.ps1
    Run in the SAME PowerShell where you will run the pipeline.

    Usage:
      Set-ExecutionPolicy -Scope Process Bypass
      .\check_onnx_cuda_env.ps1

    Optional:
      $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
      .\check_onnx_cuda_env.ps1
#>

$ErrorActionPreference = "Continue"

function H1($t){ Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function OK($t){ Write-Host "[OK]  $t" -ForegroundColor Green }
function NO($t){ Write-Host "[ERR] $t" -ForegroundColor Red }
function INF($t){ Write-Host "[..] $t" -ForegroundColor Yellow }

H1 "PowerShell / Conda env"
INF "CONDA_PREFIX = $env:CONDA_PREFIX"
INF "PWD         = $((Get-Location).Path)"
if (-not $env:CONDA_PREFIX) { INF "Conda env not detected. (Did you run: conda activate torch201 ?)"} else { OK "Conda env detected." }

H1 "NVIDIA driver (nvidia-smi)"
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvsmi) {
  NO "nvidia-smi not found in PATH."
} else {
  try {
    $out = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
      OK "nvidia-smi works."
      $out | Select-Object -First 4 | ForEach-Object { Write-Host $_ }
    } else {
      NO "nvidia-smi failed:"
      Write-Host $out
    }
  } catch { NO $_.Exception.Message }
}

H1 "CUDA toolkit path"
if ($env:CUDA_PATH) { INF "CUDA_PATH = $env:CUDA_PATH" } else { INF "CUDA_PATH is not set." }

# Try to auto-detect v12.2 if CUDA_PATH is not set
if (-not $env:CUDA_PATH) {
  $cand = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
  if (Test-Path $cand) {
    $env:CUDA_PATH = $cand
    INF "Auto-detected CUDA_PATH = $env:CUDA_PATH"
  }
}

if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
  OK "CUDA toolkit folder exists."
  $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
  if (Test-Path $nvcc) {
    try { & $nvcc --version | Select-Object -Last 5 | ForEach-Object { Write-Host $_ } } catch {}
  } else {
    INF "nvcc.exe not found under CUDA_PATH (toolkit components may be missing)."
  }
} else {
  NO "CUDA toolkit folder not found. Set CUDA_PATH to your CUDA 12.x toolkit (recommended v12.2)."
}

H1 "PATH sanity (CUDA entries)"
# Show any CUDA-related PATH entries
$cudaPathEntries = $env:PATH -split ';' | Where-Object { $_ -match "NVIDIA GPU Computing Toolkit\\CUDA" -or $_ -match "\\CUDA\\v" }
if ($cudaPathEntries.Count -gt 0) {
  INF "CUDA-related PATH entries:"
  $cudaPathEntries | ForEach-Object { Write-Host "  $_" }
} else {
  INF "No CUDA toolkit entries found in PATH."
}

H1 "Critical CUDA 12 DLLs (required by onnxruntime-gpu CUDA EP)"
$dlls = @(
  "cublasLt64_12.dll",
  "cublas64_12.dll",
  "cudart64_12.dll",
  "cufft64_12.dll",
  "cudnn64_9.dll"
)

foreach ($d in $dlls) {
  $res = & where.exe $d 2>$null
  if ($LASTEXITCODE -eq 0 -and $res) { OK "$d -> $($res[0])" } else { NO "$d not found in PATH" }
}

H1 "MSVC Runtime (common ORT requirement)"
# Check for VC runtime DLLs in System32
$vc = @("vcruntime140.dll","msvcp140.dll","vcruntime140_1.dll")
foreach ($v in $vc) {
  $p = Join-Path $env:WINDIR "System32\$v"
  if (Test-Path $p) { OK "$v present" } else { INF "$v not found in System32 (install Microsoft Visual C++ Redistributable 2015-2022 x64 if ORT complains)." }
}

H1 "Python: Torch CUDA + cuDNN"
try {
  $py = @'
import torch
print("torch.__version__      :", torch.__version__)
print("torch.version.cuda     :", torch.version.cuda)
print("torch.cuda.is_available :", torch.cuda.is_available())
print("cudnn available        :", torch.backends.cudnn.is_available())
print("cudnn enabled          :", torch.backends.cudnn.enabled)
print("cudnn version          :", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("device                :", torch.cuda.get_device_name(0))
'@
  python - <<PY
$py
PY
  OK "Torch CUDA/cuDNN check executed."
} catch {
  NO "Torch check failed: $($_.Exception.Message)"
}

H1 "Python: ONNX Runtime providers"
try {
  python - <<'PY'
import onnxruntime as ort
print("onnxruntime            :", ort.__version__)
print("available providers     :", ort.get_available_providers())
PY
  OK "ONNX Runtime provider check executed."
} catch {
  NO "ONNX Runtime check failed: $($_.Exception.Message)"
}

H1 "Actionable next steps"
INF "If you see [ERR] cublasLt64_12.dll missing:"
INF "  1) Ensure CUDA 12.2 toolkit is installed."
INF "  2) Add to PATH (for this shell):"
INF '     $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"'
INF '     $env:PATH="$env:CUDA_PATH\bin;$env:PATH"'
INF "  3) Re-run this script."
INF "If you see [ERR] cudnn64_9.dll missing:"
INF "  Install cuDNN 9.x for CUDA 12 and copy cudnn64_9.dll into CUDA\\v12.2\\bin (or add cuDNN bin to PATH)."
INF "If ORT providers include CUDAExecutionProvider but script still fails, patch make_session to prefer TensorRT."
