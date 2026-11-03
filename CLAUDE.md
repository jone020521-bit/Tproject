# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace overview

This folder is a local workspace synced to a **Jetson Nano** (NVIDIA embedded board) over SFTP. There is no build system or test suite.

## camera_app

Flask web app to turn a camera ON/OFF, stream it (MJPEG), and verify the camera is recognized. Run on the Jetson:

```
cd ~/camera_app
pip3 install flask            # never pip-install opencv-python: it lacks GStreamer; use JetPack's cv2
python3 app.py                # web UI at http://<jetson-ip>:5000
python3 app.py --check        # headless: detect + test each camera, exit code 1 on failure
python3 app.py --flip 2       # rotate CSI image 180°; also --width/--height/--fps/--port
```

- `camera.py`: `detect_cameras()` scans `/dev/video*` + sysfs `name`; `vi-output` in the name ⇒ CSI (opened via `nvarguscamerasrc` GStreamer pipeline, `sensor_id` = order among CSI nodes), otherwise USB (OpenCV `CAP_V4L2`). `CameraController` owns one active capture in a background thread; the thread itself calls `cap.release()` (nvargus locks the sensor if not released — fix with `sudo systemctl restart nvargus-daemon`).
- "Recognized" = device detected, opened, and a frame received (live: last frame < 2 s old). A device already running can't be reopened, so `check()` reports live status for it instead.
- Must stay Python 3.6 compatible (JetPack 4.x): no dataclasses, walrus, or `from __future__ import annotations`.

## Remote sync (important)

`.vscode/sftp.json` configures the VS Code SFTP extension:

- Target: `nano8@192.168.0.62:22`, remote path `/home/nano8`
- `uploadOnSave: true` — **every file saved in this folder is immediately uploaded to the Jetson's home directory**, overwriting the remote copy. Treat local edits as edits to the device.
- The local folder root maps to `/home/nano8`, so a local `foo/bar.py` lands at `/home/nano8/foo/bar.py`.

Implications:

- Code is ultimately run on the Jetson (Linux, ARM64/aarch64, JetPack/CUDA), not on this Windows machine. Don't assume Windows-only paths, tools, or x86-only packages work at runtime.
- Running or testing code requires executing on the device (e.g. `ssh nano8@192.168.0.62 '<command>'`), not locally.
- Avoid creating scratch or temporary files in this folder — they will be pushed to the device.
