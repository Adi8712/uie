import cv2
import torch

# from .original import SS_UIE_model as model
from .model import model

SIZE = 256
# WGT = "models/original.pth"
WGT = "models/weights.pth"


class Engine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.h = self.w = SIZE
        self.model = model(
            # in_channels=3, channels=16, num_resblock=4, num_memblock=4
        ).to(self.device)
        self.model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(WGT, map_location=self.device).items()
            },
            strict=False,
        )
        self.model.eval()
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def process(self, frame, profile=False):
        h, w = frame.shape[:2]

        frame = cv2.resize(frame, (self.w, self.h))

        x = torch.from_numpy(frame).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = x[:, [2, 1, 0]].mul_(1.0 / 255.0)

        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            if profile:
                self.starter.record()

            y = self.model(x)

            if profile:
                self.ender.record()

        y = y[0].permute(1, 2, 0)
        y = (y * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        y = cv2.resize(y, (w, h))

        if profile:
            torch.cuda.synchronize()
            dt = self.starter.elapsed_time(self.ender) / 1000.0
            return y[:, :, ::-1], dt

        return y[:, :, ::-1]
