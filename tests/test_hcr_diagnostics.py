import torch
from scripts.diagnose_hcr import (
    RiskStats, compare_pixel_labels, empty_transition, add_transitions,
    complete_transition, predictions_from_output
)


def test_pixel_changes_respect_ignored_gt():
    gt=torch.tensor([[[0,1,2,255,3,0]]])
    before=torch.tensor([[[1,1,2,0,0,0]]])
    after=torch.tensor([[[0,2,1,1,3,0]]])
    x=compare_pixel_labels(gt,before,after,4)
    assert (x["valid_pixels"],x["changed"],x["corrected"],x["damaged"])==(5,4,2,2)
    assert x["corrected_per_class"]==[1,0,0,1]
    assert x["damaged_per_class"]==[0,1,1,0]
    acc=empty_transition(4)
    add_transitions(acc,x)
    assert complete_transition(acc)["net_correct"]==0


def test_predictions_reproduce_hcr_mixture():
    logits=torch.tensor([[[[3.]],[[1.]]]])
    h=torch.tensor([[[[.02]],[[.98]]]])
    mixture=.5*logits.softmax(1)+.5*h
    preds=predictions_from_output({"flat_logits":logits,"hier_probs":h,
                                   "logits":mixture.log()},.5)
    assert torch.equal(preds["mixed"],preds["final"])


def test_risk_ranking():
    r=RiskStats()
    risk=torch.tensor([[[[-5.,5.]]]])
    gt=torch.tensor([[[0,0]]])
    flat=torch.tensor([[[0,1]]])
    final=torch.tensor([[[0,0]]])
    r.add(risk,flat,gt,flat,final)
    s=r.report()
    assert s["native_resolution_nodes"]==2
    assert s["auroc_approx"]>.99
    assert s["auprc_approx"]>.99
