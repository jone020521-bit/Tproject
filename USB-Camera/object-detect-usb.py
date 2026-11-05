#!/usr/bin/env python3
#
#  USB Camera - Object Detection
#
#  Real-time COCO object detection using SSD-MobileNet v2 and the OpenCV DNN module.
#  Inference runs on the CPU in its own thread, so the video stays smooth and the
#  boxes update at whatever rate the detector manages.
#
#  Press ESC or 'q' in the window to quit.
#

import argparse
import os
import sys
import threading
import time
import traceback
from collections import Counter

import cv2

WINDOW_TITLE = "USB Camera - Object Detection"
MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", MODEL_NAME)
STATS_INTERVAL = 5.0  # seconds between console status lines

# TensorFlow COCO label map. Ids are not contiguous: 80 classes across ids 1-90.
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def load_model():
    weights = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")
    config = os.path.join(MODEL_DIR, "graph.pbtxt")
    for path in (weights, config):
        if not os.path.isfile(path):
            sys.exit("Model file not found: {}\nSee the object detection section of README.md "
                     "for the download commands.".format(path))
    return cv2.dnn.readNetFromTensorflow(weights, config)


def detect(net, frame, input_size, threshold):
    """Return a list of (label, confidence, (x1, y1, x2, y2)) in frame pixels."""
    height, width = frame.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(frame, size=(input_size, input_size), swapRB=True))
    detections = []
    for _, class_id, confidence, x1, y1, x2, y2 in net.forward()[0, 0]:
        if confidence < threshold:
            continue
        label = COCO_LABELS.get(int(class_id), "id {}".format(int(class_id)))
        box = (max(int(x1 * width), 0), max(int(y1 * height), 0),
               min(int(x2 * width), width - 1), min(int(y2 * height), height - 1))
        detections.append((label, float(confidence), box))
    return detections


def draw_detections(frame, detections):
    for label, confidence, (x1, y1, x2, y2) in detections:
        # Derive the color from the label so it stays stable between updates
        seed = sum(ord(c) for c in label)
        color = (seed * 47 % 156 + 100, seed * 83 % 156 + 100, seed * 131 % 156 + 100)
        text = "{} {:.0f}%".format(label, confidence * 100)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the label inside the frame when the box touches the top edge
        text_y = max(y1, text_h + baseline + 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, text_y - text_h - baseline - 2), (x1 + text_w + 2, text_y),
                      color, cv2.FILLED)
        cv2.putText(frame, text, (x1 + 1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)


class DetectorThread(threading.Thread):
    """Runs inference on the newest frame, skipping any frames it was too slow to see."""

    def __init__(self, net, input_size, threshold):
        super().__init__(daemon=True)
        self.net = net
        self.input_size = input_size
        self.threshold = threshold
        self.condition = threading.Condition()
        self.frame = None
        self.detections = []
        self.completed = 0  # finished inferences, used for the FPS readout
        self.running = True
        self.failed = False

    def submit(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify()

    def run(self):
        try:
            while True:
                with self.condition:
                    while self.running and self.frame is None:
                        self.condition.wait()
                    if not self.running:
                        return
                    frame, self.frame = self.frame, None
                self.detections = detect(self.net, frame, self.input_size, self.threshold)
                self.completed += 1
        except Exception:
            traceback.print_exc()
            self.failed = True


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection for a USB camera")
    parser.add_argument("--device", default="/dev/video0", help="camera device (default: %(default)s)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="minimum confidence to show, 0-1 (default: %(default)s)")
    parser.add_argument("--size", type=int, default=300,
                        help="network input size; smaller is faster but misses small objects "
                             "(default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    net = load_model()

    # Leave one core free so that capture and display keep up during inference
    cv2.setNumThreads(max((os.cpu_count() or 2) - 1, 1))

    video_capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not video_capture.isOpened():
        sys.exit("Unable to open camera: {}".format(args.device))

    detector = DetectorThread(net, args.size, args.conf)
    detector.start()
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        frames = 0
        last_completed = 0
        last_stats = time.time()
        detect_fps = 0.0
        while not detector.failed:
            ret_val, frame = video_capture.read()
            if not ret_val:
                print("Lost the camera: failed to read a frame", file=sys.stderr)
                break
            detector.submit(frame.copy())
            detections = detector.detections
            draw_detections(frame, detections)
            cv2.putText(frame, "detect {:.1f} FPS".format(detect_fps), (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Under GTK+ (Jetson default) WND_PROP_AUTOSIZE goes negative once the window is closed
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            cv2.imshow(WINDOW_TITLE, frame)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code == 27 or key_code == ord('q'):
                break

            frames += 1
            now = time.time()
            if now - last_stats >= STATS_INTERVAL:
                elapsed = now - last_stats
                detect_fps = (detector.completed - last_completed) / elapsed
                counts = Counter(label for label, _, _ in detections)
                print("video {:.1f} FPS | detect {:.1f} FPS | {}".format(
                    frames / elapsed, detect_fps,
                    ", ".join("{} x{}".format(k, v) for k, v in counts.most_common()) or "nothing"),
                    flush=True)
                frames = 0
                last_completed = detector.completed
                last_stats = now
    finally:
        detector.stop()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
