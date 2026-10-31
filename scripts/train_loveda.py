#!/usr/bin/env python3
"""Train SDR-MambaNet on LoveDA Kaggle: Train/Train/{Urban,Rural}, Val/Val/{Urban,Rural}.
LoveDA IDs 1..7 -> 0..6; 0/255 ignored. 512 and 1024 tiles supported.
"""
import argparse, json, random, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sdr_mambanet import SDRConfig, SDRMambaNet
from sdr_mambanet.losses.segmentation import SegmentationLoss

ROOT = "/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation"
CLASSES = ("background","building","road","water","barren","forest","agriculture")
MEAN = np.array([.485,.456,.406],np.float32).reshape(1,1,3)
STD  = np.array([.229,.224,.225],np.float32).reshape(1,1,3)
EXTS = {".png",".jpg",".jpeg",".tif",".tiff"}

def find_pairs(root, split):
    names = ("Train","train") if split=="train" else ("Val","val","Validation")
    parents = [root/n for n in names]+[root/n/m for n in names for m in names]
    for parent in parents:
        pairs=[]
        for scene in ("Rural","Urban"):
            folder=parent/scene
            ids=next((folder/n for n in ("images_png","images") if (folder/n).is_dir()),None)
            mds=next((folder/n for n in ("masks_png","masks","labels") if (folder/n).is_dir()),None)
            if ids is None or mds is None: continue
            masks={p.stem.lower():p for p in mds.iterdir() if p.is_file() and p.suffix.lower() in EXTS}
            for p in sorted(ids.iterdir()):
                if p.suffix.lower() not in EXTS: continue
                if p.stem.lower() not in masks:
                    raise FileNotFoundError(f"No mask for {p} under {mds}")
                pairs.append((p,masks[p.stem.lower()],scene))
        if pairs:
            print(f"{split}: {len(pairs)} images (Rural={sum(z[2]=='Rural' for z in pairs)}, Urban={sum(z[2]=='Urban' for z in pairs)})",flush=True)
            return pairs
    raise FileNotFoundError(f"Unable to find image/mask pairs in {root}: expected Train/Train or Val/Val, Rural/Urban, images_png/masks_png")

def labels(pil):
    x=np.asarray(pil)
    if x.ndim!=2: raise ValueError("LoveDA masks must have grayscale/indexed IDs, not RGB colors.")
    if not np.isin(x,[0,1,2,3,4,5,6,7,255]).all():
        raise ValueError(f"Unexpected LoveDA IDs: {np.unique(x).tolist()}")
    return np.where((x>=1)&(x<=7),x.astype(np.int16)-1,255).astype(np.uint8)

class LoveDA(Dataset):
    def __init__(self, pairs, size, train):
        self.pairs,self.size,self.train=pairs,size,train
        self.tiles=[]
        if not train:
            for i,(path,_,_) in enumerate(pairs):
                with Image.open(path) as im: w,h=im.size
                self.tiles += [(i,x,y) for y in range(0,h,size) for x in range(0,w,size)]
    def __len__(self): return len(self.pairs) if self.train else len(self.tiles)
    def __getitem__(self,j):
        if self.train: i=j
        else: i,x,y=self.tiles[j]
        ip,mp,_=self.pairs[i]
        with Image.open(ip) as im: img=im.convert("RGB")
        with Image.open(mp) as im: mask=im.copy()
        if img.size!=mask.size: raise ValueError(f"Size mismatch: {ip} and {mp}")
        w,h=img.size
        s=self.size
        if self.train:
            x=random.randint(0,max(0,w-s)); y=random.randint(0,max(0,h-s))
        img=np.asarray(img.crop((x,y,x+s,y+s)),np.uint8)
        mask=labels(mask.crop((x,y,x+s,y+s)))
        mask[max(0,h-y):,:]=255
        mask[:,max(0,w-x):]=255
        if self.train:
            if random.random()<.5: img,mask=np.flip(img,0),np.flip(mask,0)
            if random.random()<.5: img,mask=np.flip(img,1),np.flip(mask,1)
            n=random.randrange(4)
            img,mask=np.rot90(img,n),np.rot90(mask,n)
        img=(np.ascontiguousarray(img,dtype=np.float32)/255-MEAN)/STD
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))),torch.from_numpy(np.ascontiguousarray(mask,dtype=np.int64))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    cm=torch.zeros((7,7),dtype=torch.int64)
    for x,y in loader:
        with torch.autocast("cuda",enabled=device.type=="cuda"):
            pred=model(x.to(device,non_blocking=True))["logits"].argmax(1).cpu()
        good=(y!=255)
        cm+=torch.bincount((y[good].long()*7+pred[good].long()),minlength=49).reshape(7,7)
    tp=cm.diag().float()
    union=cm.sum(0).float()+cm.sum(1).float()-tp
    iou=tp/union.clamp_min(1)
    return dict(miou=float(iou[union>0].mean()),oa=float(tp.sum()/cm.sum().clamp_min(1)),
                iou_per_class={n:round(float(iou[i]),5) for i,n in enumerate(CLASSES)},confusion=cm.tolist())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=ROOT)
    p.add_argument("--size",type=int,choices=(512,1024),default=512)
    p.add_argument("--context",choices=("conv","mamba"),default="conv")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--accum",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--output",default="/kaggle/working/sdr_loveda")
    p.add_argument("--resnet-weights-path",default=None)
    p.add_argument("--no-pretrained",action="store_true")
    p.add_argument("--resume",default=None)
    a=p.parse_args()
    if a.batch<1 or a.accum<1: p.error("--batch and --accum must be positive")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=Path(a.root)
    train=LoveDA(find_pairs(root,"train"),a.size,True)
    val=LoveDA(find_pairs(root,"val"),a.size,False)
    dlcfg=dict(num_workers=a.workers,pin_memory=(dev.type=="cuda"))
    train_dl=DataLoader(train,batch_size=a.batch,shuffle=True,**dlcfg)
    val_dl=DataLoader(val,batch_size=1,shuffle=False,**dlcfg)
    cfg=SDRConfig(num_classes=7,semantic_encoder="resnet34",pretrained_semantic=not a.no_pretrained,
                  semantic_weights_path=a.resnet_weights_path,context_type=a.context)
    try: model=SDRMambaNet(cfg).to(dev)
    except Exception:
        print("Check ResNet34 ImageNet weights (Kaggle Internet/local checkpoint), torchvision and optional mamba_ssm installation.",flush=True)
        raise
    loss_fn=SegmentationLoss(ignore_index=255,dice_weight=1.).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/"args.json").write_text(json.dumps(vars(a),indent=2))
    start,best=0,-1.
    if a.resume:
        z=torch.load(a.resume,map_location="cpu",weights_only=False)
        model.load_state_dict(z["model"])
        optimizer.load_state_dict(z["optimizer"])
        scheduler.load_state_dict(z["scheduler"])
        scaler.load_state_dict(z["scaler"])
        start,best=z["epoch"]+1,z["best_miou"]
    print(f"GPU={dev}, size={a.size}, context={a.context}, val tiles={len(val)}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",flush=True)
    for epoch in range(start,a.epochs):
        model.train();optimizer.zero_grad(set_to_none=True);running=0.
        for step,(x,y) in enumerate(train_dl):
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            group_start=(step//a.accum)*a.accum
            group_size=min(a.accum,len(train_dl)-group_start)
            with torch.autocast("cuda",enabled=dev.type=="cuda"):
                loss=loss_fn(model(x),y)["total"]
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at epoch={epoch+1}, step={step}")
            scaler.scale(loss/group_size).backward()
            running+=float(loss.detach())
            if (step+1)%a.accum==0 or step+1==len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        stats=validate(model,val_dl,dev)
        stats.update(epoch=epoch+1,train_loss=running/len(train_dl))
        print(f"Epoch {epoch+1}/{a.epochs} train_loss={stats['train_loss']:.4f} val_mIoU={stats['miou']:.4f} OA={stats['oa']:.4f}",flush=True)
        with (out/"history.jsonl").open("a") as f: f.write(json.dumps(stats)+"\n")
        if stats["miou"]>best:
            best=stats["miou"]
            torch.save(dict(model=model.state_dict(),epoch=epoch,best_miou=best,model_config=asdict(cfg),
                            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict()),out/"best.pt")
            (out/"best_metrics.json").write_text(json.dumps(stats,indent=2))
        torch.save(dict(model=model.state_dict(),epoch=epoch,best_miou=best,model_config=asdict(cfg),
                        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict()),out/"last.pt")

if __name__=="__main__": main()
