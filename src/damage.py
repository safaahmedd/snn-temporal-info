import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from src.data import CachedFrames
from src.model import VGGSNN

FIELDS = ['dataset', 'T', 'seed', 'damage', 'level', 'effective', 'acc']


def truncate(x, frac):
    """Keep the first frac of the time bins, blank the rest."""
    T = x.shape[1]
    keep = int(np.ceil(frac * T - 1e-9))
    x = x.clone()
    x[:, keep:] = 0
    return x, keep / T


def drop_frames(x, p, gen):
    """Blank each frame independently with probability p."""
    N, T = x.shape[:2]
    keep = (torch.rand(N, T, generator=gen, device=x.device) >= p).to(x.dtype)
    return x * keep.view(N, T, 1, 1, 1), 1 - p


def jitter_frames(x, sigma_ms, window_ms, gen):
    """Shift each frame in time by a random offset (sigma in milliseconds).
    Offsets smaller than half a frame have no effect (frame-level approximation)."""
    N, T = x.shape[:2]
    slice_ms = window_ms / T
    shift = torch.round(torch.randn(N, T, generator=gen, device=x.device)
                        * sigma_ms / slice_ms).long()
    idx = (torch.arange(T, device=x.device).view(1, T) + shift).clamp(0, T - 1)
    idx = idx.view(N, T, 1, 1, 1).expand(-1, -1, *x.shape[2:])
    return torch.gather(x, 1, idx), (shift != 0).float().mean().item()


@torch.no_grad()
def accuracy(net, loader, damage_fn, device):
    net.eval()
    correct = total = 0
    effs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if damage_fn is not None:
            x, e = damage_fn(x)
            effs.append(e)
        out = net(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        functional.reset_net(net)
    eff = float(np.mean(effs)) if effs else ''
    return correct / total, eff


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['dataset'], int(r['T']), int(r['seed']), r['damage'], r['level'])
                for r in csv.DictReader(f)}


def damage_sweep(cfg, Ts, seeds, ckpt_dir, out_csv,
                 trunc=(0.75, 0.5, 0.25), drops=(0.1, 0.2, 0.3, 0.4, 0.5),
                 jitters=(25, 50, 100, 200), batch_size=64):
    device = 'cuda'
    name = cfg['name']
    window_ms = cfg['window_us'] / 1000
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = done_keys(out_csv)

    for seed in seeds:
        for T in Ts:
            path = f'{ckpt_dir}/{name}_T{T}_seed{seed}.pt'
            if not os.path.exists(path):
                print('missing checkpoint:', path)
                continue
            ds = CachedFrames(name, 'test', T, cfg['window_us'], cfg['size'],
                              cfg['cache_dir'], binary=cfg['binary'])
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            net = VGGSNN(num_classes=cfg.get('num_classes', 11), size=cfg['size']).to(device)
            net.load_state_dict(torch.load(path, map_location=device))

            def gen():
                g = torch.Generator(device=device)
                g.manual_seed(1000 + seed)
                return g

            jobs = [('none', 0.0, None)]
            for f in trunc:
                if int(np.ceil(f * T - 1e-9)) < T:          # skip if nothing is removed
                    jobs.append(('truncation', f, lambda x, f=f: truncate(x, f)))
            for p in drops:
                jobs.append(('frame_drop', p, lambda x, p=p, g=gen(): drop_frames(x, p, g)))
            for s in jitters:
                jobs.append(('jitter_ms', s,
                             lambda x, s=s, g=gen(): jitter_frames(x, s, window_ms, g)))

            for dmg, level, fn in jobs:
                key = (name, T, seed, dmg, str(level))
                if key in done:
                    continue
                acc, eff = accuracy(net, loader, fn, device)
                new_file = not os.path.exists(out_csv)
                with open(out_csv, 'a', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if new_file:
                        w.writeheader()
                    w.writerow({'dataset': name, 'T': T, 'seed': seed, 'damage': dmg,
                                'level': level, 'effective': eff, 'acc': acc})
                print(f'T={T} seed={seed} {dmg} {level}: acc {acc:.4f}')
            torch.cuda.empty_cache()