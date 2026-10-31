"""CPU smoke tests for layer4 and bidirectional branch exchange."""
import copy
import torch
from sdr_mambanet import SDRConfig, SDRMambaNet


def _model():
    return SDRMambaNet(SDRConfig(
        semantic_encoder="resnet34", pretrained_semantic=False, context_type="conv"))


def test_stage_shapes_and_gradients():
    torch.manual_seed(7)
    model = _model().train()
    image = torch.randn(1, 3, 128, 128)
    maps = model.backbone(image)
    assert maps["d4"].shape == (1, 48, 32, 32)
    assert maps["d8"].shape == (1, 96, 16, 16)
    assert maps["s8"].shape == (1, 128, 16, 16)
    assert maps["s16"].shape == (1, 160, 8, 8)
    assert maps["s32"].shape == (1, 160, 4, 4)
    logits = model(image)["logits"]
    assert logits.shape == (1, 7, 128, 128)
    logits.mean().backward()
    assert model.backbone.bi8.to_semantic.weight.grad is not None
    assert model.backbone.bi8.to_detail.weight.grad is not None
    assert model.backbone.cross16.from_detail.weight.grad is not None
    assert model.backbone.semantic.layer4[0].conv1.weight.grad is not None


def test_deploy_reparam_equivalence():
    model = _model().eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = model(image)["logits"]
        deploy = copy.deepcopy(model).eval().switch_to_deploy()
        after = deploy(image)["logits"]
    assert (before - after).abs().max().item() < 1e-4
