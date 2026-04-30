import pathlib
import sys
import time

import cv2

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from inference.engine import Engine

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

engine = Engine()

vid_path = DATA_DIR / "input.mp4"
# out_path = DATA_DIR / "original.mp4"
out_path = DATA_DIR / "new.mp4"

cap = cv2.VideoCapture(str(vid_path))

if not cap.isOpened():
    raise RuntimeError("Can't open video")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

if not out.isOpened():
    print("avc1 failed, falling back to mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

start = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = engine.process(frame)
    out.write(processed)
    frames += 1

cap.release()
out.release()

taken = time.time() - start
print(f"Processed {frames} frames in {taken:.4f}s ({frames / taken:.4f} FPS)")
