#!/usr/bin/env python3
#
#  Check the labels, report per-class counts, and write the Darknet training lists:
#  train.txt, valid.txt, and obj.data. All paths are relative to this folder, so the
#  whole folder can be copied to a faster machine for training.
#
#  Photos that have not been labeled yet are left out and reported.
#

import argparse
import glob
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_BOXES_PER_CLASS = 200  # below this, warn that the class is likely too small to learn well


def check_label_file(path, class_count):
    """Return (class ids in the file, list of problems found)."""
    class_ids, problems = [], []
    with open(path) as label_file:
        for number, line in enumerate(label_file, 1):
            if not line.strip():
                continue
            where = "{}:{}".format(os.path.basename(path), number)
            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError
                class_id = int(fields[0])
                cx, cy, w, h = [float(field) for field in fields[1:]]
            except ValueError:
                problems.append("{}: expected 'class cx cy w h'".format(where))
                continue
            if not 0 <= class_id < class_count:
                problems.append("{}: class {} is out of range".format(where, class_id))
            elif not (w > 0 and h > 0 and cx - w / 2 >= -1e-3 and cx + w / 2 <= 1 + 1e-3
                      and cy - h / 2 >= -1e-3 and cy + h / 2 <= 1 + 1e-3):
                problems.append("{}: box is outside the image".format(where))
            else:
                class_ids.append(class_id)
    return class_ids, problems


def main():
    parser = argparse.ArgumentParser(description="Validate labels and write Darknet training lists")
    parser.add_argument("--root", default=HERE, help="folder holding obj.names and dataset/images")
    parser.add_argument("--val", type=float, default=0.2, help="validation share (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed, for a repeatable split")
    args = parser.parse_args()

    with open(os.path.join(args.root, "obj.names")) as names_file:
        names = [line.strip() for line in names_file if line.strip()]

    images = sorted(glob.glob(os.path.join(args.root, "dataset", "images", "*.jpg")))
    labeled, unlabeled, problems = [], 0, []
    labels_by_image = {}
    for image in images:
        path = os.path.splitext(image)[0] + ".txt"
        if not os.path.exists(path):
            unlabeled += 1
            continue
        class_ids, file_problems = check_label_file(path, len(names))
        problems.extend(file_problems)
        labeled.append(image)
        labels_by_image[image] = class_ids

    if problems:
        print("\n".join(problems))
        sys.exit("{} label problem(s). Fix them and run this again.".format(len(problems)))
    if len(labeled) < 2:
        sys.exit("Need at least 2 labeled images, found {}.".format(len(labeled)))

    random.Random(args.seed).shuffle(labeled)
    val_count = max(int(len(labeled) * args.val), 1)
    valid, train = labeled[:val_count], labeled[val_count:]

    def boxes_in(subset):
        return Counter(c for image in subset for c in labels_by_image[image])

    train_boxes, valid_boxes = boxes_in(train), boxes_in(valid)
    print("{:<18}{:>8}{:>8}".format("class", "train", "valid"))
    warnings = []
    for class_id, name in enumerate(names):
        print("{:<18}{:>8}{:>8}".format(name, train_boxes[class_id], valid_boxes[class_id]))
        if train_boxes[class_id] < MIN_BOXES_PER_CLASS:
            warnings.append("{}: only {} training boxes (aim for {}+)".format(
                name, train_boxes[class_id], MIN_BOXES_PER_CLASS))
        if not valid_boxes[class_id]:
            warnings.append("{}: no validation boxes, so its accuracy cannot be measured".format(name))
    background = sum(1 for image in labeled if not labels_by_image[image])
    print("\n{} train, {} valid, {} background, {} not labeled yet (left out)".format(
        len(train), len(valid), background, unlabeled))
    for warning in warnings:
        print("WARNING " + warning)

    def write_list(file_name, subset):
        with open(os.path.join(args.root, file_name), "w") as list_file:
            for image in subset:
                list_file.write(os.path.relpath(image, args.root) + "\n")

    write_list("train.txt", train)
    write_list("valid.txt", valid)
    with open(os.path.join(args.root, "obj.data"), "w") as data_file:
        data_file.write("classes = {}\ntrain = train.txt\nvalid = valid.txt\n"
                        "names = obj.names\nbackup = backup/\n".format(len(names)))
    os.makedirs(os.path.join(args.root, "backup"), exist_ok=True)


if __name__ == "__main__":
    main()
