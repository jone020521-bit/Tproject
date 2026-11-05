#!/usr/bin/env python3
#
#  Draw YOLO bounding boxes on the collected photos.
#
#  Left drag   draw a box with the current class
#  Right click delete the box under the pointer
#  1-7         choose the class
#  d / SPACE   next image        a  previous image
#  u           undo last box     e  mark as background (no objects in the photo)
#  ESC / q     save and quit
#
#  Labels are saved automatically, as a .txt next to each image, whenever you change image.
#

import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW_TITLE = "Label images"
PANEL_WIDTH = 210
MIN_BOX_PIXELS = 4
COLORS = [(0, 200, 255), (60, 140, 220), (255, 255, 255), (80, 220, 80),
          (0, 140, 255), (255, 180, 60), (40, 40, 255)]


def label_path(image_path):
    return os.path.splitext(image_path)[0] + ".txt"


def read_labels(path, width, height):
    """Read a YOLO label file into [class, x1, y1, x2, y2] boxes in pixels."""
    boxes = []
    with open(path) as label_file:
        for line in label_file:
            if line.strip():
                class_id, cx, cy, w, h = line.split()
                cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
                boxes.append([int(class_id), round(cx - w / 2), round(cy - h / 2),
                              round(cx + w / 2), round(cy + h / 2)])
    return boxes


def format_labels(boxes, width, height):
    lines = []
    for class_id, x1, y1, x2, y2 in boxes:
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
            class_id, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
            (x2 - x1) / width, (y2 - y1) / height))
    return "".join(lines)


class Labeler:
    def __init__(self, image_paths, class_names):
        self.paths = image_paths
        self.names = class_names
        self.current_class = 0
        self.drag_start = None
        self.pointer = (0, 0)
        self.open_image(self.first_unlabeled())

    def first_unlabeled(self):
        for index, path in enumerate(self.paths):
            if not os.path.exists(label_path(path)):
                return index
        return 0

    def open_image(self, index):
        self.index = index
        path = self.paths[index]
        self.image = cv2.imread(path)
        if self.image is None:
            sys.exit("Could not read image: {}".format(path))
        self.height, self.width = self.image.shape[:2]
        self.boxes = []
        self.background = False
        self.drag_start = None
        if os.path.exists(label_path(path)):
            self.boxes = read_labels(label_path(path), self.width, self.height)
            self.background = not self.boxes

    def save(self):
        """Write the labels. A photo with no boxes stays unlabeled unless marked as background."""
        path = label_path(self.paths[self.index])
        if self.boxes or self.background:
            with open(path, "w") as label_file:
                label_file.write(format_labels(self.boxes, self.width, self.height))
        elif os.path.exists(path):
            os.remove(path)

    def go(self, step):
        self.save()
        self.open_image(min(max(self.index + step, 0), len(self.paths) - 1))

    def clamp(self, x, y):
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)

    def on_mouse(self, event, x, y, _flags, _param):
        x, y = self.clamp(x, y)
        self.pointer = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.drag_start = None
            if abs(x - x0) >= MIN_BOX_PIXELS and abs(y - y0) >= MIN_BOX_PIXELS:
                self.boxes.append([self.current_class, min(x, x0), min(y, y0), max(x, x0), max(y, y0)])
                self.background = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            # With overlapping boxes, the smallest one is the one the user is pointing at
            hits = [b for b in self.boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
            if hits:
                self.boxes.remove(min(hits, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])))

    def on_key(self, key_code):
        """Handle a key press. Returns False when the tool should quit."""
        if key_code == 27 or key_code == ord('q'):
            self.save()
            return False
        if ord('1') <= key_code < ord('1') + len(self.names):
            self.current_class = key_code - ord('1')
        elif key_code == ord('d') or key_code == ord(' '):
            self.go(1)
        elif key_code == ord('a'):
            self.go(-1)
        elif key_code == ord('u') and self.boxes:
            self.boxes.pop()
        elif key_code == ord('e') and not self.boxes:
            self.background = not self.background
        return True

    def render(self):
        view = self.image.copy()
        for class_id, x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(view, (x1, y1), (x2, y2), COLORS[class_id], 2)
            cv2.putText(view, self.names[class_id], (x1 + 2, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[class_id], 1, cv2.LINE_AA)
        if self.drag_start:
            cv2.rectangle(view, self.drag_start, self.pointer, COLORS[self.current_class], 1)
        if self.background:
            cv2.putText(view, "BACKGROUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        panel = np.full((self.height, PANEL_WIDTH, 3), 40, dtype=np.uint8)
        lines = ["{} / {}".format(self.index + 1, len(self.paths)), ""]
        for line_index, text in enumerate(lines):
            cv2.putText(panel, text, (10, 22 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)
        for class_id, name in enumerate(self.names):
            y = 66 + 24 * class_id
            if class_id == self.current_class:
                cv2.rectangle(panel, (4, y - 16), (PANEL_WIDTH - 4, y + 6), COLORS[class_id], 2)
            cv2.putText(panel, "{} {}".format(class_id + 1, name), (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, COLORS[class_id], 1, cv2.LINE_AA)
        return np.hstack((view, panel))


def main():
    parser = argparse.ArgumentParser(description="Draw YOLO bounding boxes on collected photos")
    parser.add_argument("--images", default=os.path.join(HERE, "dataset", "images"),
                        help="image folder (default: dataset/images)")
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if not image_paths:
        sys.exit("No .jpg images in {}. Capture some with collect.py first.".format(args.images))
    with open(os.path.join(HERE, "obj.names")) as names_file:
        class_names = [line.strip() for line in names_file if line.strip()]

    print(__doc__)
    labeler = Labeler(image_paths, class_names)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_TITLE, labeler.on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_TITLE, labeler.render())
            key_code = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                labeler.save()
                break
            if key_code != 255 and not labeler.on_key(key_code):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
