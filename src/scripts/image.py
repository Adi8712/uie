import pathlib
import sys

import cv2

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from inference.engine import Engine

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

engine = Engine()

img_path = DATA_DIR / "test.jpg"
out_path = DATA_DIR / "result.jpg"

img = cv2.imread(str(img_path))
out = engine.process(img)

cv2.imwrite(str(out_path), out)
