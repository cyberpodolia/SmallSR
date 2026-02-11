param(
    [string]$Image = "project_clean:ngc-trt",
    [string]$ProjectRoot = $(Split-Path -Parent $PSScriptRoot),
    [string]$EnginePath = "/app/weights/student_sr_1280x960_fp16_trt107_linux.engine",
    [int]$Port = 8001,
    [string]$Display = "host.docker.internal:0.0"
)

$ErrorActionPreference = "Stop"
chcp 65001 | Out-Null

$CamUrl = "http://host.docker.internal:$Port/stream.mjpeg"

docker run --rm -it --gpus all `
  -e DISPLAY=$Display `
  -e QT_X11_NO_MITSHM=1 `
  -e CAM_URL=$CamUrl `
  -v "${ProjectRoot}:/app" `
  -w /app/python `
  $Image `
  python realtime_trt_student.py `
    --engine-path $EnginePath `
    --source http `
    --cam-url $CamUrl
