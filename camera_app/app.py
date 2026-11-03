"""Jetson Nano 카메라 제어 웹 애플리케이션.

실행:  python3 app.py            -> 브라우저에서 http://<jetson-ip>:5000
점검:  python3 app.py --check    -> 웹 서버 없이 카메라 감지/인식 결과만 출력
"""
import argparse
import sys
import time

from flask import Flask, Response, jsonify, render_template, request

from camera import CameraController, detect_cameras

app = Flask(__name__)
controller = None


def _error(message, code=400):
    return jsonify({"ok": False, "error": message}), code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras")
def api_cameras():
    return jsonify({"ok": True, "cameras": detect_cameras()})


@app.route("/api/status")
def api_status():
    return jsonify({"ok": True, "status": controller.status()})


@app.route("/api/camera/on", methods=["POST"])
def api_on():
    device = (request.get_json(silent=True) or {}).get("device")
    if not device:
        cams = detect_cameras()
        if not cams:
            return _error("감지된 카메라가 없습니다", 404)
        device = cams[0]["device"]
    try:
        controller.start(device)
    except RuntimeError as e:
        return _error(str(e), 500)
    return jsonify({"ok": True, "status": controller.status()})


@app.route("/api/camera/off", methods=["POST"])
def api_off():
    controller.stop()
    return jsonify({"ok": True, "status": controller.status()})


@app.route("/api/check", methods=["POST"])
def api_check():
    device = (request.get_json(silent=True) or {}).get("device")
    if not device:
        return _error("device 값이 필요합니다")
    return jsonify({"ok": True, "result": controller.check(device)})


@app.route("/stream.mjpg")
def stream():
    if not controller.running:
        return _error("카메라가 꺼져 있습니다", 409)

    def generate():
        interval = 1.0 / max(controller.fps, 1)
        while controller.running:
            jpg = controller.get_jpeg()
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(interval)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot():
    jpg = controller.get_jpeg(quality=95)
    if jpg is None:
        return _error("카메라가 꺼져 있습니다", 409)
    return Response(jpg, mimetype="image/jpeg")


def run_check():
    cams = detect_cameras()
    if not cams:
        print("감지된 카메라가 없습니다. (/dev/video* 없음)")
        print("- CSI: 리본 케이블 방향/고정 확인 후 재부팅")
        print("- USB: 'lsusb' 로 장치 인식 여부 확인")
        return 1

    failed = 0
    for cam in cams:
        print("[%s] %s (%s)" % (cam["device"], cam["name"], cam["type"]))
        r = controller.check(cam["device"])
        if r["frame_ok"]:
            print("  -> 인식 정상: %s, 밝기 %.1f" % (r["resolution"], r["brightness"]))
            if r.get("warning"):
                print("  !  " + r["warning"])
        else:
            failed += 1
            print("  -> 인식 실패: %s" % r.get("error", "알 수 없는 오류"))
    return 1 if failed else 0


def main():
    global controller
    parser = argparse.ArgumentParser(description="Jetson Nano 카메라 제어")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=int, default=0, help="CSI flip-method (0~7, 2=180도 회전)")
    parser.add_argument("--check", action="store_true", help="카메라 인식만 점검하고 종료")
    args = parser.parse_args()

    controller = CameraController(args.width, args.height, args.fps, args.flip)

    if args.check:
        sys.exit(run_check())

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
