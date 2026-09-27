"""Run from the repo root BEFORE any training:  python check_ablation.py"""
import torch, torch.nn.functional as F
from models.model_stages import BoundaryRefine, BRH_VARIANTS
from loss.loss import BoundaryOhemCELoss, OhemCELoss

torch.manual_seed(0)
dev = 'cuda'
l8 = torch.randn(2, 19, 32, 64, device=dev)       # stride-8 logits
hr = torch.randn(2, 64, 64, 128, device=dev)      # stride-4 feat_res4
print("variant      out shape         params  in_optim  init==baseline")
for v in BRH_VARIANTS:
    b = BoundaryRefine(variant=v).to(dev).train()
    o = b(l8, hr)
    wd, nwd = b.get_params()
    in_optim = set(map(id, b.parameters())) == set(map(id, wd + nwd))
    up = F.interpolate(l8, o.shape[2:], mode='bilinear', align_corners=True)
    print("%-12s %-17s %6d  %-8s  %s" % (v, tuple(o.shape), sum(p.numel() for p in b.parameters()),
                                         in_optim, torch.allclose(o, up)))

lab = torch.randint(0, 19, (2, 256, 256), device=dev); lab[:, :20] = 255
lg = torch.randn(2, 19, 256, 256, device=dev)
n_min = 2 * 256 * 256 // 16
print("\nstock OhemCELoss        %.6f" % OhemCELoss(0.7, n_min)(lg, lab).item())
for mode in ('weighted_rank', 'unweighted_rank', 'none'):
    for w in (1.0, 3.0):
        print("%-16s w=%.0f   %.6f" % (mode, w, BoundaryOhemCELoss(0.7, n_min, w_bnd=w, ohem_mode=mode)(lg, lab).item()))
print("plain CE                %.6f" % torch.nn.CrossEntropyLoss(ignore_index=255)(lg, lab).item())