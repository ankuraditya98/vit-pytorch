import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
v.eval()
img = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    preds = v(img) # (1, 1000)
print(preds.shape)
