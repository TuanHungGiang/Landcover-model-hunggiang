"""CPU smoke tests for the first complete HCR method (no dataset required)."""
import copy
import torch
from sdr_mambanet.hcr_model import HCRConfig, HCRSDRNet, build_confusion_groups
from sdr_mambanet.losses.hcr import HCRSegmentationLoss


def _small_model():
    return HCRSDRNet(HCRConfig(pretrained_semantic=False, context_type="conv",
                                graph_steps=2))


def test_train_confusion_groups_are_symmetric_frequency_normalized():
    cm = torch.tensor([[80, 20, 0, 0], [15, 80, 5, 0],
                       [0, 0, 80, 20], [0, 0, 18, 82]])
    mapping, groups = build_confusion_groups(cm)
    assert len(groups) == 2
    assert mapping[0] == mapping[1]
    assert mapping[2] == mapping[3]
    assert mapping[0] != mapping[2]


def test_trainable_hcr_forward_loss_and_gradient():
    torch.manual_seed(3)
    model = _small_model().train()
    cm = torch.eye(7, dtype=torch.long) * 80
    cm[0, 6] = 30
    cm[6, 0] = 20
    groups = model.configure_groups_from_training(cm)
    assert any({0, 6} == set(g) for g in groups)
    x = torch.randn(1, 3, 64, 64)
    y = torch.randint(0, 7, (1, 64, 64), dtype=torch.long)
    y[:, :5, :5] = 255
    out = model(x)
    assert out["logits"].shape == (1, 7, 64, 64)
    assert out["risk_logits"].shape == (1, 1, 8, 8)
    assert out["hier_probs"].shape == (1, 7, 8, 8)
    torch.testing.assert_close(out["hier_probs"].sum(1), torch.ones(1, 8, 8),
                               atol=1e-5, rtol=0)
    assert torch.isfinite(out["logits"]).all()
    losses = HCRSegmentationLoss()(out, y)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    for p in (model.group_head.weight, model.fine_head[-1].weight,
              model.graph_refiner.risk_head[-1].weight,
              model.graph_refiner.semantic_proj.weight,
              model.graph_refiner.detail_proj.weight,
              model.backbone.detail.stage8[0].spatial.branches["region"][1].weight):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@torch.no_grad()
def test_checkpoint_roundtrip_and_reparam():
    torch.manual_seed(5)
    model = _small_model().eval()
    cm = torch.eye(7, dtype=torch.long) * 50
    cm[2, 3] = 12
    model.configure_groups_from_training(cm)
    image = torch.randn(1, 3, 64, 64)
    a = model(image)["logits"]
    restored = _small_model().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(image)["logits"], a, rtol=0, atol=1e-5)
    deploy = copy.deepcopy(restored).eval().switch_to_deploy()
    torch.testing.assert_close(deploy(image)["logits"], a, rtol=0, atol=3e-4)


def test_hierarchy_independent_of_class_names():
    model = HCRSDRNet(HCRConfig(num_classes=6, pretrained_semantic=False))
    assert model.group_head.out_channels == 6
    assert model.class_groups.numel() == 6



def test_memory_saver_checkpoint_backward_and_bn_stats():
    # All main-method paths remain trainable; ResNet and RSF branch BatchNorm
    # update running stats ONCE despite recomputation in backward.
    torch.manual_seed(19)
    model = HCRSDRNet(HCRConfig(pretrained_semantic=False, memory_saver=True,
                                graph_steps=2)).train()
    semantic_bn = model.backbone.semantic.layer1[0].bn1
    before = semantic_bn.running_mean.clone()
    seen_before = int(semantic_bn.num_batches_tracked.item())
    rsf_bn = model.backbone.detail.stage4[0].spatial.branches["region"][1]
    rsf_seen_before = int(rsf_bn.num_batches_tracked.item())
    x = torch.randn(1, 3, 64, 64)
    target = torch.randint(0, 7, (1, 64, 64))
    out = model(x)
    loss = HCRSegmentationLoss()(out, target)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert semantic_bn.training and rsf_bn.training
    assert int(semantic_bn.num_batches_tracked.item()) == seen_before + 1
    assert int(rsf_bn.num_batches_tracked.item()) == rsf_seen_before + 1
    assert not torch.equal(semantic_bn.running_mean, before)
    assert model.backbone.semantic.layer3[0].conv1.weight.grad is not None
    assert model.graph_refiner.detail_proj.weight.grad is not None
    assert model.backbone.detail.stage4[0].spatial.branches["region"][1].weight.grad is not None

def test_v2_rejects_one_way_confusion_and_ramps_hierarchy():
    conf = torch.eye(7,dtype=torch.long)*100
    conf[3,6]=70
    mapping,groups=build_confusion_groups(conf,min_score=0.02,min_reciprocal=0.015)
    assert mapping[3]!=mapping[6], "One-way confusion must not create pair"
    model=HCRSDRNet(HCRConfig(pretrained_semantic=False,graph_steps=1))
    assert model.cfg.architecture_version=="r34_layer4_bi_hcr_v2"
    assert float(model.hierarchy_strength)==0.
    model.set_hierarchy_strength(0.75)
    assert abs(float(model.hierarchy_strength)-0.75)<1e-6
    cm=torch.eye(7,dtype=torch.long)*100
    cm[0,4]=30
    cm[4,0]=28
    groups=model.configure_groups_from_training(cm)
    assert any(set(g)=={0,4} for g in groups)
