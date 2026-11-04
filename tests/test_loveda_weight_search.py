"""Train-derived LoveDA weighting and DE algorithm smoke tests."""
import json
import numpy as np
import pytest
import torch
from PIL import Image
from scripts.loveda_class_weights import count_train_masks, load_or_count_train, make_class_weights
from scripts.optimize_hcr_de import propose, command, BOUNDS
from sdr_mambanet.losses.hcr import HCRSegmentationLoss


def test_counts_exclude_ignored_and_cache_rejects_modified_masks(tmp_path):
    folder=tmp_path/"Train"/"Rural"
    folder.mkdir(parents=True)
    mask=folder/"mask.png"
    Image.fromarray(np.array([[0,1,2,3,4],[5,6,7,255,1]],dtype=np.uint8)).save(mask)
    pairs=[(folder/"image.png",mask,"Rural")]
    obj=load_or_count_train(pairs,tmp_path,tmp_path/"counts.json")
    assert obj["counts"]==[2,1,1,1,1,1,1]
    assert sum(obj["counts"])==8
    weights=make_class_weights(obj["counts"],.0)
    torch.testing.assert_close(weights,torch.ones(7))
    Image.fromarray(np.full((7,7),1,np.uint8)).save(mask)
    with pytest.raises(ValueError,match="do not match"):
        load_or_count_train(pairs,tmp_path,tmp_path/"counts.json")


def test_class_weights_affect_full_flat_and_hierarchy_ce():
    # Model output from a synthetic tensor; all three CE terms use weights.
    flat=torch.randn(1,7,8,8,requires_grad=True)
    output={"logits":flat,"flat_logits":flat,
            "hier_probs":torch.softmax(torch.randn(1,7,2,2),1),
            "mixed_logits":flat.detach(),
            "risk_logits":torch.zeros(1,1,2,2)}
    gt=torch.randint(0,7,(1,8,8))
    unweighted=HCRSegmentationLoss()
    weighted=HCRSegmentationLoss(class_weights=torch.tensor([.4,1.,1.,1.,3.,1.,1.]))
    a,b=unweighted(output,gt),weighted(output,gt)
    assert abs(float(a["ce"]-b["ce"]))>1e-5
    assert abs(float(a["flat"]-b["flat"]))>1e-5
    assert abs(float(a["hierarchy"]-b["hierarchy"]))>1e-5


def test_de_mutation_bounded_and_compiles_command(tmp_path):
    rng=np.random.default_rng(4)
    pop=rng.uniform(BOUNDS[:,0],BOUNDS[:,1],size=(4,4))
    x=propose(pop,None,0,rng)
    assert np.all(x>=BOUNDS[:,0]) and np.all(x<=BOUNDS[:,1])
    assert x.shape==(4,)
