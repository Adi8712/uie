import pathlib
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from inference.engine import Engine

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

engine = Engine()

vid_path = str(DATA_DIR / "input.mp4")
# out_path = str(DATA_DIR / "original.mp4")
out_path = str(DATA_DIR / "new.mp4")

cap = cv2.VideoCapture(vid_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

decoder = subprocess.Popen(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-vsync",
        "0",
        "-i",
        vid_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ],
    stdout=subprocess.PIPE,
    bufsize=10**8,
)

encoder = subprocess.Popen(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        out_path,
    ],
    stdin=subprocess.PIPE,
)

frame_size = w * h * 3

decode_q = queue.Queue(maxsize=32)
encode_q = queue.Queue(maxsize=32)


def reader():
    while True:
        raw = decoder.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        decode_q.put(frame)
    decode_q.put(None)


def writer():
    while True:
        item = encode_q.get()
        if item is None:
            break
        encoder.stdin.write(item)
        encode_q.task_done()


t1 = threading.Thread(target=reader)
t2 = threading.Thread(target=writer)

t1.start()
t2.start()

frames = 0

torch.cuda.synchronize()
start = time.time()

timer = 0.0

while True:
    frame = decode_q.get()
    if frame is None:
        break

    y, dt = engine.process(frame, profile=True)
    timer += dt

    encode_q.put(y.tobytes())
    frames += 1

encode_q.put(None)

t1.join()
t2.join()

decoder.stdout.close()
encoder.stdin.close()

decoder.wait()
encoder.wait()

torch.cuda.synchronize()

taken = time.time() - start
print(f"Processed {frames} frames in {taken:.4f}s")
print(f"Model FPS: {frames / timer:.4f}")
print(f"Pipeline FPS: {frames / taken:.4f}")
