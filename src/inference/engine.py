import cv2
import torch

from .model import model

SIZE = 256
WGT = "models/weights.pth"


class Engine:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.h = self.w = SIZE
        self.model = model().to(self.device)
        _state = torch.load(WGT, map_location=self.device)
        self.model.load_state_dict(_state)
        self.model.eval()

    def process(self, frame):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (self.w, self.h)), cv2.COLOR_BGR2RGB)

        x = torch.from_numpy(frame_rgb).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            y = self.model(x).squeeze(0)
            y = torch.clamp(y, 0, 1)
            y = (y.permute(1, 2, 0) * 255).byte().cpu().numpy()

        out = cv2.cvtColor(cv2.resize(y, (w, h)), cv2.COLOR_RGB2BGR)

        return out
