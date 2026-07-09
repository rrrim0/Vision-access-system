import torch
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained="vggface2").eval()
torch.save(model.state_dict(), "models/facenet_vggface2.pt")

print("Сохранено: models/facenet_vggface2.pt")