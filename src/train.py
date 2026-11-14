import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, neuron

from src.utils import set_seed
from src.data import CachedFrames
from src.model import VGGSNN


def evaluate(net, loader, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            functional.reset_net(net)
    return correct / total


def measure_firing_rate(net, loader, device):
    """Share of neuron-timesteps that spike, over the whole test set."""
    net.eval()
    stats = {'spikes': 0.0, 'count': 0}

    def hook(m, inp, out):
        stats['spikes'] += out.sum().item()
        stats['count'] += out.numel()

    hooks = [m.register_forward_hook(hook)
             for m in net.modules() if isinstance(m, neuron.LIFNode)]
    with torch.no_grad():
        for x, y in loader:
            net(x.to(device))
            functional.reset_net(net)
    for h in hooks:
        h.remove()
    return stats['spikes'] / stats['count']


def measure_latency(net, ds, device, n_warm=10, n_meas=100):
    """Milliseconds per sample, one sample at a time."""
    net.eval()
    xs = [ds[i][0].unsqueeze(0).to(device) for i in range(n_warm + n_meas)]
    with torch.no_grad():
        for i in range(n_warm):
            net(xs[i]); functional.reset_net(net)
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_warm, n_warm + n_meas):
            net(xs[i]); functional.reset_net(net)
        torch.cuda.synchronize()
    return (time.time() - t0) / n_meas * 1000


def run(cfg):
    set_seed(cfg['seed'])
    device = 'cuda'

    def make(split):
        return CachedFrames(cfg['name'], split, cfg['T'], cfg['window_us'],
                            cfg['size'], cfg['cache_dir'], binary=cfg['binary'])

    train_loader = DataLoader(make('train'), batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(make('test'), batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=2)

    net = VGGSNN(num_classes=cfg.get('num_classes', 11), size=cfg['size']).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    acc = 0.0
    for ep in range(cfg['epochs']):
        net.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            functional.reset_net(net)
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        acc = evaluate(net, test_loader, device)
        print(f"T={cfg['T']} seed={cfg['seed']} epoch {ep + 1}/{cfg['epochs']}  "
              f"loss {total_loss / n:.3f}  test acc {acc:.4f}  "
              f"elapsed {time.time() - t0:.0f}s")

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**20

    if cfg.get('ckpt_dir'):
        os.makedirs(cfg['ckpt_dir'], exist_ok=True)
        torch.save(net.state_dict(),
                   f"{cfg['ckpt_dir']}/{cfg['name']}_T{cfg['T']}_seed{cfg['seed']}.pt")

    return {'dataset': cfg['name'], 'T': cfg['T'], 'seed': cfg['seed'],
            'epochs': cfg['epochs'], 'window_us': cfg['window_us'],
            'size': cfg['size'], 'binary': cfg['binary'],
            'final_acc': acc,                         # last epoch, not the best
            'firing_rate': measure_firing_rate(net, test_loader, device),
            'latency_ms': measure_latency(net, test_loader.dataset, device),
            'train_time_s': round(train_time, 1),
            'peak_mem_MB': round(peak_mem, 1)}