"""Safe conversion/checkpoint utility.

The original model must be in eval mode; conversion is destructive.
Keep a copy of the training checkpoint for future fine-tuning.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
from pathlib import Path
import torch
from ..models import SDRMambaNet


@torch.no_grad()
def verify_deploy_equivalence(model: SDRMambaNet, shape=(1, 3, 128, 128),
                              atol: float = 1e-4, rtol: float = 1e-4):
    model = copy.deepcopy(model).eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.randn(*shape, device=device, dtype=dtype)
    reference = model(x)["logits"]
    model.switch_to_deploy()
    deployed = model(x)["logits"]
    diff = (reference.float() - deployed.float()).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "allclose": bool(torch.allclose(reference, deployed, atol=atol, rtol=rtol)),
    }


def save_deploy_checkpoint(model: SDRMambaNet, filepath: str | Path,
                           remove_aux: bool = True) -> None:
    deployed = copy.deepcopy(model).eval().switch_to_deploy(remove_aux=remove_aux)
    torch.save({
        "config": asdict(deployed.cfg),
        "state_dict": deployed.state_dict(),
        "deploy": True,
        "note": "RSFRepDW train-time branches folded; context/fusion unchanged.",
    }, str(filepath))


def load_deploy_checkpoint(filepath: str | Path, device: str = "cpu") -> SDRMambaNet:
    """Rebuild and load a folded inference model from save_deploy_checkpoint."""
    from ..models import SDRConfig
    obj = torch.load(str(filepath), map_location=device, weights_only=False)
    if not obj.get("deploy"):
        raise ValueError("Not a deploy checkpoint")
    settings = dict(obj["config"])
    settings["pretrained_semantic"] = False  # never download just to load a checkpoint
    model = SDRMambaNet(SDRConfig(**settings)).to(device).eval()
    model.switch_to_deploy(remove_aux=True)
    model.load_state_dict(obj["state_dict"], strict=True)
    return model