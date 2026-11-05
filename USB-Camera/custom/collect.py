#!/usr/bin/env python3
#
#  Capture training photos from the USB camera.
#
#  In the camera window: SPACE saves the current frame, ESC or 'q' quits.
#  In the terminal (for SSH sessions): Enter saves, 'q' then Enter quits.
#

import argparse
import datetime
import glob
import os
import queue
import sys
import threading
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW_TITLE = "Collect training images"
MIN_SAVE_INTERVAL = 0.5  # seconds; holding SPACE repeats the key and would save near-identical frames


def read_terminal(commands):
    for line in sys.stdin:
        commands.put(line.strip().lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="camera device (default: %(default)s)")
    parser.add_argument("--out", default=os.path.join(HERE, "dataset", "images"),
                        help="output folder (default: dataset/images)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    saved = len(glob.glob(os.path.join(args.out, "*.jpg")))

    video_capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not video_capture.isOpened():
        sys.exit("Unable to open camera: {} (is another camera app still running?)".format(args.device))

    commands = queue.Queue()
    threading.Thread(target=read_terminal, args=(commands,), daemon=True).start()
    print("Camera window: SPACE saves, q quits. Terminal: Enter saves, q then Enter quits.", flush=True)

    last_save = 0.0
    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        while True:
            ret_val, frame = video_capture.read()
            if not ret_val:
                sys.exit("Lost the camera: failed to read a frame")
            # The counter is drawn on a copy so that it never ends up in the saved photo
            preview = frame.copy()
            cv2.putText(preview, "saved: {}  [SPACE] save  [q] quit".format(saved), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            cv2.imshow(WINDOW_TITLE, preview)
            key_code = cv2.waitKey(10) & 0xFF
            quit_requested = key_code == 27 or key_code == ord('q')
            save_requested = key_code == ord(' ')
            while not commands.empty():
                if commands.get() == "q":
                    quit_requested = True
                else:
                    save_requested = True
            if quit_requested:
                break
            if save_requested and time.time() - last_save >= MIN_SAVE_INTERVAL:
                name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                if cv2.imwrite(os.path.join(args.out, name), frame):
                    saved += 1
                    last_save = time.time()
                    print("saved {}: {}".format(saved, name), flush=True)
                else:
                    sys.exit("Could not write to {}".format(args.out))
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
    print("{} images in {}".format(saved, args.out))


if __name__ == "__main__":
    main()
