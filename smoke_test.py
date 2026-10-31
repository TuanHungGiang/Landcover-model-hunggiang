import torch
from sdr_mambanet import SDRConfig, SDRMambaNet

model = SDRMambaNet(SDRConfig(semantic_encoder="resnet34", pretrained_semantic=False, context_type="conv"))
model.eval()
with torch.no_grad():
    result = model(torch.randn(1,3,128,128))
print({k:tuple(v.shape) for k,v in result.items()})
