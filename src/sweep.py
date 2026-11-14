import os, csv
import torch
from src.train import run

FIELDS = ['dataset', 'T', 'seed', 'epochs', 'window_us', 'size', 'binary',
          'final_acc', 'firing_rate', 'latency_ms', 'train_time_s', 'peak_mem_MB']


def done_runs(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r['dataset'], int(r['T']), int(r['seed'])) for r in csv.DictReader(f)}


def sweep(base_cfg, Ts, seeds, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = done_runs(out_csv)
    for seed in seeds:                 # all T for seed 0 first, then seed 1, ...
        for T in Ts:
            key = (base_cfg['name'], T, seed)
            if key in done:
                print('skip (already done):', key)
                continue
            print('running:', key)
            res = run(dict(base_cfg, T=T, seed=seed))
            new_file = not os.path.exists(out_csv)
            with open(out_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow(res)
            print('saved:', res)
            torch.cuda.empty_cache()