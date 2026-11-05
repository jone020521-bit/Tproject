#!/usr/bin/env python3
#
#  Temporary demo: find the wood cube's printed face without any trained model.
#
#  This is NOT the YOLO detector. It matches ORB image features against the printed
#  face in the labeled photos, so it only knows this one object and only sees the
#  face that was photographed.
#
#  Press ESC or 'q' in the window to quit.
#

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW_TITLE = "Wood cube demo (feature matching, not YOLO)"
BOX_CLASS_ID = 1
TEMPLATE_COUNT = 4
CENTER_FRACTION = 0.7  # template uses the middle of the box, which keeps fingers out of it
RATIO_TEST = 0.75
MIN_INLIERS = 10
STATS_INTERVAL = 5.0


def read_box_labels(image_path, width, height):
    boxes = []
    with open(os.path.splitext(image_path)[0] + ".txt") as label_file:
        for line in label_file:
            fields = line.split()
            if len(fields) == 5 and int(fields[0]) == BOX_CLASS_ID:
                cx, cy, w, h = [float(field) for field in fields[1:]]
                boxes.append((cx * width, cy * height, w * width, h * height))
    return boxes


class CubeFinder:
    def __init__(self, image_paths):
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        candidates = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape
            for cx, cy, w, h in read_box_labels(path, width, height):
                half_w, half_h = w * CENTER_FRACTION / 2, h * CENTER_FRACTION / 2
                x0, y0 = int(max(cx - half_w, 0)), int(max(cy - half_h, 0))
                crop = image[y0:int(cy + half_h), x0:int(cx + half_w)]
                keypoints, descriptors = self.orb.detectAndCompute(crop, None)
                if descriptors is not None and len(keypoints) >= MIN_INLIERS:
                    candidates.append((crop.shape, keypoints, descriptors))
        if not candidates:
            sys.exit("No usable labeled photos of the wood cube were found.")
        # The best template from each evenly spaced group, so the set spans the shooting session
        groups = np.array_split(np.arange(len(candidates)), min(TEMPLATE_COUNT, len(candidates)))
        self.templates = [candidates[max(group, key=lambda i: len(candidates[i][1]))] for group in groups]

    def find(self, gray):
        """Return the best cube outline as a 4x2 array of pixels, or None."""
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return None
        best_outline, best_inliers = None, MIN_INLIERS - 1
        for (height, width), template_points, template_descriptors in self.templates:
            pairs = self.matcher.knnMatch(template_descriptors, descriptors, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
            if len(good) < MIN_INLIERS:
                continue
            source = np.float32([template_points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            target = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if matrix is None or int(mask.sum()) <= best_inliers:
                continue
            # Grow the template corners back out to the full face of the cube
            margin_x = width * (1 / CENTER_FRACTION - 1) / 2
            margin_y = height * (1 / CENTER_FRACTION - 1) / 2
            corners = np.float32([[-margin_x, -margin_y], [width + margin_x, -margin_y],
                                  [width + margin_x, height + margin_y], [-margin_x, height + margin_y]])
            outline = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            if self.plausible(outline):
                best_outline, best_inliers = outline, int(mask.sum())
        return best_outline

    @staticmethod
    def plausible(outline):
        """Reject the twisted or squashed shapes that a false match produces."""
        contour = outline.astype(np.int32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 900:
            return False
        sides = [np.linalg.norm(outline[i] - outline[(i + 1) % 4]) for i in range(4)]
        return max(sides) / max(min(sides), 1.0) < 2.5


def main():
    parser = argparse.ArgumentParser(description="Temporary wood cube demo using feature matching")
    parser.add_argument("--device", default="/dev/video0", help="camera device (default: %(default)s)")
    args = parser.parse_args()

    finder = CubeFinder(sorted(glob.glob(os.path.join(HERE, "dataset", "images", "*.jpg"))))
    video_capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not video_capture.isOpened():
        sys.exit("Unable to open camera: {} (is another camera app still running?)".format(args.device))

    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        frames = found = 0
        last_stats = time.time()
        while True:
            ret_val, frame = video_capture.read()
            if not ret_val:
                sys.exit("Lost the camera: failed to read a frame")
            outline = finder.find(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if outline is not None:
                found += 1
                points = outline.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                x, y = points[:, 0, 0].min(), points[:, 0, 1].min()
                cv2.putText(frame, "wood cube", (int(max(x, 0)), int(max(y - 6, 14))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            cv2.imshow(WINDOW_TITLE, frame)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code == 27 or key_code == ord('q'):
                break
            frames += 1
            now = time.time()
            if now - last_stats >= STATS_INTERVAL:
                print("{:.1f} FPS | cube found in {}/{} frames".format(
                    frames / (now - last_stats), found, frames), flush=True)
                frames = found = 0
                last_stats = now
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
