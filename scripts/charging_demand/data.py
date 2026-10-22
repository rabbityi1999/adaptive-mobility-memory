"""Regional hourly forecasting data, reconstructed from completed events only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

CHANNELS = ['order_count', 'start_cohort_energy_kwh']
INPUT_FILES = ['PROTOCOL.json', 'series.npz', 'events.npz', 'windows.csv',
               'normalization_train_only.npz', 'VERIFIED.json']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


class Data:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {name: digest(self.root / name) for name in INPUT_FILES}
        verified = json.loads((self.root / 'VERIFIED.json').read_text(encoding='utf-8'))
        assert verified['passed']
        for name in INPUT_FILES:
            if name in verified['artifact_sha256']:
                assert self.hashes[name] == verified['artifact_sha256'][name], name
        with np.load(self.root / 'series.npz', allow_pickle=False) as z:
            self.ids = np.flatnonzero(z['core_mask'])
            self.full_regions = z['demand'].shape[1]
            self.y = z['demand'][:, self.ids].copy()
            self.calendar = z['calendar'].astype(np.float32)
            self.timestamps = z['timestamps'].copy()
        self.windows = pd.read_csv(self.root / 'windows.csv')
        self.events = dict(np.load(self.root / 'events.npz', allow_pickle=False))
        self.e = self.events
        mapping = np.full(self.full_regions, -1, dtype=np.int64)
        mapping[self.ids] = np.arange(len(self.ids))
        core = mapping[self.e['region_index']] >= 0
        self.starts = self.e['start_seconds'][core]
        self.ends = self.e['end_seconds'][core]
        self.regions = mapping[self.e['region_index'][core]]
        self.energy = self.e['energy_kwh'][core]
        self.order = np.argsort(self.ends, kind='stable')
        ts = pd.to_datetime(self.timestamps)
        self.weekhour = (ts.dayofweek * 24 + ts.hour).to_numpy()
        self.hour = ts.hour.to_numpy()
        self.weekday = ts.dayofweek.to_numpy()
        self.train_end = 1080
        avail = self.aggregate(self.ends < self.train_end * 3600)[:self.train_end]
        self.region_mean = avail.mean(axis=0).astype(np.float32)
        self.region_std = np.maximum(avail.std(axis=0), [1.0, 20.0]).astype(np.float32)
        self.channel_std = np.maximum(avail.std(axis=(0, 1)), [1.0, 20.0]).astype(np.float32)
        self.week_means = np.stack([avail[self.weekhour[:self.train_end] == k].mean(0)
                                    for k in range(168)]).astype(np.float32)

    def aggregate(self, mask):
        result = np.zeros_like(self.y)
        pos = (self.starts[mask] // 3600, self.regions[mask])
        np.add.at(result[:, :, 0], pos, 1)
        np.add.at(result[:, :, 1], pos, self.energy[mask])
        return result

    def history(self, origin):
        mask = ((self.starts >= (origin - 168) * 3600) &
                (self.starts < origin * 3600) & (self.ends <= origin * 3600))
        return self.aggregate(mask)[origin - 168:origin].astype(np.float32)

    def split(self, split):
        rows = self.windows[self.windows.split == split]
        origins = rows.origin_index.to_numpy(dtype=np.int64)
        assert len(origins) == {'train': 865, 'validation': 337, 'test': 265}[split]
        bounds = {'train': (0, 1080), 'validation': (1080, 1464), 'test': (1464, 1776)}[split]
        assert (origins >= max(bounds[0], 168)).all()
        assert (origins + 48 <= bounds[1]).all()
        x = np.empty((len(origins), 168, len(self.ids), 2), dtype=np.float32)
        y = np.empty((len(origins), 24, len(self.ids), 2), dtype=np.float32)
        past = np.empty((len(origins), 168, 5), dtype=np.float32)
        future = np.empty((len(origins), 24, 5), dtype=np.float32)
        available = np.zeros_like(self.y)
        cursor = 0
        for i, q in enumerate(origins):
            stop = np.searchsorted(self.ends[self.order], q * 3600, side='right')
            new = self.order[cursor:stop]
            pos = (self.starts[new] // 3600, self.regions[new])
            np.add.at(available[:, :, 0], pos, 1)
            np.add.at(available[:, :, 1], pos, self.energy[new])
            cursor = stop
            x[i] = available[q - 168:q]
            y[i] = self.y[q:q + 24]
            past[i] = self.calendar[q - 168:q]
            future[i] = self.calendar[q:q + 24]
        # Compare streaming implementation against independent event masks.
        for i in (0, len(origins) // 2, len(origins) - 1):
            np.testing.assert_array_equal(x[i], self.history(int(origins[i])))
        return {'x': x, 'y': y, 'past': past, 'future': future, 'origins': origins}

    def baselines(self, bundle):
        target = bundle['origins'][:, None] + np.arange(24)
        return {'zero': np.zeros_like(bundle['y']),
                'previous_day_asof': bundle['x'][:, -24:],
                'previous_week_asof': bundle['x'][:, :24],
                'training_hour_of_week_mean': self.week_means[self.weekhour[target]]}


def metrics(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    assert y.shape == p.shape and np.isfinite(p).all() and (p >= 0).all()
    out = {}
    for c, name in enumerate(CHANNELS):
        a, b = y[..., c], p[..., c]
        def calc(t, v):
            err = v - t
            nz = t > 0
            return {'MAE': float(np.abs(err).mean()), 'RMSE': float(np.sqrt((err ** 2).mean())),
                    'WAPE': float(np.abs(err).sum() / t.sum()),
                    'positive_target_MAE': float(np.abs(err)[nz].mean()),
                    'bias': float(err.mean())}
        out[name] = {'all_24_steps': calc(a, b),
                     'lead_hours': {str(k): calc(a[:, k-1], b[:, k-1])
                                    for k in (1, 3, 6, 12, 24)},
                     'region_MAE': np.abs(b-a).mean(axis=(0, 1)).tolist()}
    return out


def balanced_score(y, p, baseline_mae):
    mae = np.abs(p.astype(np.float64) - y).mean(axis=(0, 1, 2))
    return float((mae / baseline_mae).mean())
