import torch

ckpt = torch.load("exp/checkpoint_epoch_0.pth", map_location="cpu")

model = ckpt["model"]

bad = 0

for name, tensor in model.items():
    if torch.is_tensor(tensor):
        if not torch.isfinite(tensor).all():
            print(name, "HAS NaN")
            bad += 1

print()
print("Bad model tensors:", bad)