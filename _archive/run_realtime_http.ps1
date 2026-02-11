$Image = "project_clean:ngc-trt"
$ProjectRoot = $PSScriptRoot
$CamUrl = "http://host.docker.internal:8000/stream.mjpeg"

docker run --rm -it --gpus all `
  -e CAM_URL=$CamUrl `
  -v "$ProjectRoot:/app" `
  -w /app/python `
  $Image `
  python RUN_REALTIME_TEST.py --source http
