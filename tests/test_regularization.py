import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.loveda_regularization import (
    RegularizedLoveDA, build_discriminative_groups, jitter_rgb,
    lr_multiplier, top_confusions,
)
from scripts.loveda_fast import FastLoveDA


def pairs(tmp_path: Path):
    rng = np.random.default_rng(123)
    image = rng.integers(0,255,(96,96,3),dtype=np.uint8)
    mask = np.full((96,96), 1, dtype=np.uint8)  # raw background->model class 0
    mask[15:40,15:40] = 5  # raw barren->model class 4
    mask[55:90,45:90] = 7  # raw agriculture->model class 6
    mask[:5,:] = 0  # ignore
    ip,mp=tmp_path/'im.png',tmp_path/'mask.png'
    Image.fromarray(image).save(ip)
    Image.fromarray(mask).save(mp)
    return [(ip,mp,'Rural')]


def test_augmented_train_does_not_corrupt_mask(tmp_path):
    ds = RegularizedLoveDA(pairs(tmp_path),64,True,cache_size=1,
                           augment='strong',focus_probability=1.0,
                           focus_classes=(4,6),zoom_min=1.2,zoom_max=1.2)
    random.seed(4)
    image,mask=ds[0]
    assert image.shape==(3,64,64) and image.dtype==torch.float32
    assert mask.shape==(64,64) and mask.dtype==torch.int64
    assert set(mask.unique().tolist()) <= {0,4,6,255}
    assert ((mask==4)|(mask==6)).any()
    assert torch.isfinite(image).all()


def test_no_aug_is_identical_to_original(tmp_path):
    ref = FastLoveDA(pairs(tmp_path), 64, True,cache_size=0)
    ds = RegularizedLoveDA(pairs(tmp_path), 64, True,cache_size=0,
                           augment='none')
    random.seed(123); a=ref[0]
    random.seed(123); b=ds[0]
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])


def test_lr_schedule_warmup_cosine():
    values=[lr_multiplier(e,60,3,0.1) for e in range(60)]
    assert values[:3]==pytest.approx([1/3,2/3,1])
    assert values[3]==pytest.approx(1)
    assert values[-1]==pytest.approx(0.1)
    assert all(v>=0.1 for v in values)
    assert all(values[e]>=values[e+1] for e in range(3,59))


def test_top_confusions_reports_gt_to_pred():
    cm=np.array([[10,20,0],[0,25,5],[3,1,50]])
    rows=top_confusions(cm,('a','b','c'),top_k=2)
    assert rows[0]['gt']=='a' and rows[0]['pred']=='b'
    assert rows[0]['percent_of_gt']==pytest.approx(66.67)


def test_discriminative_groups_cover_unique_params():
    class TinySemantic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k in ('stem','layer1','layer2','layer3','layer4'):
                setattr(self,k,torch.nn.Conv2d(2,2,1))
            self.context=torch.nn.Conv2d(2,2,1)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Module()
            self.backbone.semantic=TinySemantic()
            self.backbone.detail=torch.nn.Conv2d(2,2,1)
            self.decoder=torch.nn.Conv2d(2,2,1)
    m=Tiny()
    groups=build_discriminative_groups(m,2e-4,5e-5)
    assert len(groups)==2 and groups[0]['lr']==5e-5 and groups[1]['lr']==2e-4
    assert len(set(id(p) for g in groups for p in g['params']))==len(list(m.parameters()))