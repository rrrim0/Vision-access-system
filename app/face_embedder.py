import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEmbedder:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=True,
            device=self.device
        )

        self.model = InceptionResnetV1(
            pretrained="vggface2",
            classify=False
        ).eval().to(self.device)

    def get_embedding(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        det = self.mtcnn.detect(rgb, landmarks=True)
        if det is None:
            return None, None

        if isinstance(det, tuple) and len(det) == 2:
            boxes, probs = det
        else:
            boxes, probs, _landmarks = det

        if boxes is None or len(boxes) == 0:
            return None, None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        box = boxes[idx]

        faces = self.mtcnn.extract(rgb, [box], save_path=None)
        if faces is None or len(faces) == 0:
            return None, None

        face_tensor = faces[0].unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(face_tensor)

        emb = emb[0].cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        x1, y1, x2, y2 = map(int, box)
        return emb, (x1, y1, x2, y2)