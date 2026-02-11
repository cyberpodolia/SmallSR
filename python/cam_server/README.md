# cam_server

Lightweight HTTP camera bridge for Windows.

## Run on Windows (host)

```powershell
python python/cam_server/run_cam_server.py --port 8000
```

Endpoints:
- `http://127.0.0.1:8000/frame.jpg`
- `http://127.0.0.1:8000/stream.mjpeg`
- `http://127.0.0.1:8000/health` (200 when frames are available)

## Run in container (client)

```bash
python /app/python/RUN_REALTIME_TEST.py --source http --cam-url http://host.docker.internal:8000/stream.mjpeg
```

Or with env var:

```bash
export CAM_URL=http://host.docker.internal:8000/stream.mjpeg
python /app/python/RUN_REALTIME_TEST.py --source http
```

## One-liner demo

```powershell
.\run_realtime_http.ps1
```
