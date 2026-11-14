import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tonic
from tqdm import tqdm


def events_to_frames(ev, in_size, out_size, T, window_us):
    """Turn one event recording into a (T, 2, out_size, out_size) count array.

    The recording is shifted to start at time 0 and cut to window_us.
    The window is split into T equal time bins (slice = window_us / T).
    Channel 0 and 1 are the two polarities.
    """
    t = ev['t'].astype(np.int64)
    t = t - t.min()
    keep = t < window_us
    t = t[keep]
    x = ev['x'][keep].astype(np.int64) * out_size // in_size
    y = ev['y'][keep].astype(np.int64) * out_size // in_size
    p = ev['p'][keep].astype(np.int64)
    b = t * T // window_us                      # time-bin index, 0..T-1
    frames = np.zeros((T, 2, out_size, out_size), dtype=np.uint16)
    np.add.at(frames, (b, p, y, x), 1)
    return frames

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

def get_raw(name, split, raw_root):
    train = (split == 'train')
    if name == 'dvsgesture':
        return DVS128Gesture(root=raw_root, train=train, data_type='event'), 128
    raise ValueError(f'Unknown dataset: {name}')


def cache_path(cache_dir, name, split, T, window_us, size):
    return f'{cache_dir}/{name}_{split}_T{T}_W{window_us}_S{size}.npz'


def build_cache(name, Ts, window_us, size, raw_root, cache_dir):
    """Convert every recording once for every T in Ts, and save to cache_dir.

    Skips any file that already exists, so it is safe to re-run after a
    disconnect.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for split in ['train', 'test']:
        todo = [T for T in Ts
                if not os.path.exists(cache_path(cache_dir, name, split, T, window_us, size))]
        if not todo:
            continue
        ds, in_size = get_raw(name, split, raw_root)
        X = {T: np.zeros((len(ds), T, 2, size, size), dtype=np.uint16) for T in todo}
        Y = np.zeros(len(ds), dtype=np.int64)
        for i in tqdm(range(len(ds)), desc=f'{name} {split}'):
            ev, label = ds[i]
            Y[i] = label
            for T in todo:
                X[T][i] = events_to_frames(ev, in_size, size, T, window_us)
        for T in todo:
            np.savez_compressed(cache_path(cache_dir, name, split, T, window_us, size),
                                X=X[T], Y=Y)


class CachedFrames(Dataset):
    """Loads the cached frames. binary=True turns counts into 0/1."""

    def __init__(self, name, split, T, window_us, size, cache_dir, binary=False):
        d = np.load(cache_path(cache_dir, name, split, T, window_us, size))
        self.X, self.Y, self.binary = d['X'], d['Y'], binary

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32))
        if self.binary:
            x = (x > 0).float()
        return x, int(self.Y[i])