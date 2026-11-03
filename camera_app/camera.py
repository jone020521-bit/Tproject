"""Jetson Nano 카메라 감지 / ON·OFF 제어 / 인식 확인 모듈.

- CSI 카메라(IMX219 등): GStreamer nvarguscamerasrc 파이프라인으로 연다.
- USB 카메라: OpenCV V4L2 백엔드로 연다.
JetPack 기본 Python 3.6 에서도 동작하도록 작성했다.
"""
import glob
import os
import re
import threading
import time

import cv2


def _read_sysfs(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (IOError, OSError):
        return ""


def detect_cameras():
    """/dev/video* 장치를 찾아 CSI/USB 를 구분한 목록을 반환한다."""
    cameras = []
    nodes = []
    for path in glob.glob("/dev/video*"):
        m = re.match(r"^/dev/video(\d+)$", path)
        if m:
            nodes.append((int(m.group(1)), path))

    csi_sensor_id = 0
    for num, path in sorted(nodes):
        sysfs = "/sys/class/video4linux/video%d" % num
        # USB 카메라는 메타데이터용 노드를 추가로 만들 수 있다. index 0 만 실제 캡처 노드.
        index = _read_sysfs(os.path.join(sysfs, "index"))
        if index and index != "0":
            continue

        name = _read_sysfs(os.path.join(sysfs, "name")) or "unknown"
        is_csi = "vi-output" in name.lower()
        cam = {
            "device": path,
            "index": num,
            "name": name,
            "type": "CSI" if is_csi else "USB",
        }
        if is_csi:
            cam["sensor_id"] = csi_sensor_id
            csi_sensor_id += 1
        cameras.append(cam)
    return cameras


def find_camera(device):
    for cam in detect_cameras():
        if cam["device"] == device:
            return cam
    return None


def csi_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1, format=NV12 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, width, height, fps, flip, width, height)
    )


def open_capture(cam, width, height, fps, flip):
    if cam["type"] == "CSI":
        cap = cv2.VideoCapture(
            csi_pipeline(cam["sensor_id"], width, height, fps, flip), cv2.CAP_GSTREAMER
        )
    else:
        cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap.release()
        hint = ""
        if cam["type"] == "CSI":
            hint = " (OpenCV GStreamer 지원 여부 확인, 또는 'sudo systemctl restart nvargus-daemon')"
        raise RuntimeError("%s 장치를 열 수 없습니다%s" % (cam["device"], hint))
    return cap


def read_first_frame(cap, timeout=5.0):
    """카메라 초기화 지연(CSI 는 1~2초)을 고려해 첫 프레임을 기다린다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.1)
    return None


def frame_brightness(frame):
    small = cv2.resize(frame, (64, 36))
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def check_camera(cam, width=1280, height=720, fps=30, flip=0, timeout=5.0):
    """장치를 잠깐 열어 프레임 수신 여부로 인식 상태를 확인한다."""
    result = dict(cam)
    result.update({"detected": True, "opened": False, "frame_ok": False})
    try:
        cap = open_capture(cam, width, height, fps, flip)
    except RuntimeError as e:
        result["error"] = str(e)
        return result
    try:
        result["opened"] = True
        frame = read_first_frame(cap, timeout)
        if frame is None:
            result["error"] = "장치는 열렸으나 %.0f초 동안 프레임을 받지 못했습니다" % timeout
            return result
        result["frame_ok"] = True
        result["resolution"] = "%dx%d" % (frame.shape[1], frame.shape[0])
        result["brightness"] = round(frame_brightness(frame), 1)
        if result["brightness"] < 5:
            result["warning"] = "영상이 거의 검정색입니다 (렌즈 캡/조명 확인)"
        return result
    finally:
        cap.release()


class CameraController(object):
    """한 번에 하나의 카메라를 ON/OFF 하고 최신 프레임을 보관한다."""

    def __init__(self, width=1280, height=720, fps=30, flip=0):
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip

        self._ctl_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._reset_state()
        self.last_error = None

    def _reset_state(self):
        self.camera = None
        self._frame = None
        self._frame_count = 0
        self._last_frame_ts = 0.0
        self._started_ts = 0.0
        self._measured_fps = 0.0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, device):
        with self._ctl_lock:
            if self.running:
                if self.camera and self.camera["device"] == device:
                    return
                self._stop_locked()

            cam = find_camera(device)
            if cam is None:
                raise RuntimeError("%s 장치가 감지되지 않습니다" % device)

            cap = open_capture(cam, self.width, self.height, self.fps, self.flip)
            frame = read_first_frame(cap)
            if frame is None:
                cap.release()
                raise RuntimeError("%s 에서 프레임을 받지 못했습니다" % device)

            self._reset_state()
            self.camera = cam
            self.last_error = None
            self._frame = frame
            self._frame_count = 1
            self._last_frame_ts = self._started_ts = time.time()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, args=(cap,))
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        with self._ctl_lock:
            self._stop_locked()

    def _stop_locked(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None
        with self._frame_lock:
            self._frame = None
        self.camera = None

    def _loop(self, cap):
        failures = 0
        window_start, window_frames = time.time(), 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 50:
                        self.last_error = "프레임 수신이 계속 실패하여 카메라를 중지했습니다"
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                now = time.time()
                with self._frame_lock:
                    self._frame = frame
                    self._frame_count += 1
                    self._last_frame_ts = now
                window_frames += 1
                if now - window_start >= 1.0:
                    self._measured_fps = window_frames / (now - window_start)
                    window_start, window_frames = now, 0
        finally:
            # nvargus 는 release 하지 않으면 다음 실행 시 장치가 잠길 수 있다.
            cap.release()

    def get_frame(self):
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def get_jpeg(self, quality=80):
        frame = self.get_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def status(self):
        running = self.running
        st = {
            "running": running,
            "camera": self.camera if running else None,
            "error": self.last_error,
        }
        if running:
            frame = self.get_frame()
            age = time.time() - self._last_frame_ts
            st.update({
                "frame_count": self._frame_count,
                "fps": round(self._measured_fps, 1),
                "uptime_sec": round(time.time() - self._started_ts, 1),
                "last_frame_age_sec": round(age, 2),
                # 최근 2초 안에 프레임이 들어왔으면 정상 인식으로 본다.
                "recognized": age < 2.0,
            })
            if frame is not None:
                st["resolution"] = "%dx%d" % (frame.shape[1], frame.shape[0])
                st["brightness"] = round(frame_brightness(frame), 1)
        return st

    def check(self, device):
        cam = find_camera(device)
        if cam is None:
            return {"device": device, "detected": False, "opened": False,
                    "frame_ok": False, "error": "장치가 감지되지 않습니다"}
        if self.running and self.camera and self.camera["device"] == device:
            # 이미 켜져 있는 장치는 다시 열 수 없으므로 실시간 상태로 판단한다.
            st = self.status()
            result = dict(cam)
            result.update({"detected": True, "opened": True,
                           "frame_ok": st.get("recognized", False),
                           "resolution": st.get("resolution"),
                           "brightness": st.get("brightness")})
            return result
        return check_camera(cam, self.width, self.height, self.fps, self.flip)
