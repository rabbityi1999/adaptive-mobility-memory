"""Compact direct multi-step baselines; no claim of published-model reproduction."""
import numpy as np
import torch
from torch import nn


class SequenceModel(nn.Module):
    def __init__(self, kind, regions, config, initial_mean):
        super().__init__()
        self.kind, self.regions = kind, regions
        n_features = regions * 2 + 6  # two demands per region + calendar5 + maturity1
        width = config['mlp_hidden_size'] if kind == 'mlp' else config['hidden_size']
        if kind == 'mlp':
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(168*n_features, width),
                                         nn.ReLU(), nn.Dropout(config['dropout']))
        elif kind in ('gru', 'lstm'):
            cls = nn.GRU if kind == 'gru' else nn.LSTM
            self.encoder = cls(n_features, width, batch_first=True)
        else:
            raise ValueError(kind)
        self.dropout = nn.Dropout(config['dropout'])
        self.head = nn.Linear(width + 24*5, 24*regions*2)
        nn.init.normal_(self.head.weight, std=0.001)
        initial = np.tile(initial_mean[None], (24, 1, 1)).reshape(-1)
        # Numerically stable inverse softplus.
        inverse = np.maximum(initial, 1e-4)
        inverse = inverse + np.log(-np.expm1(-inverse))
        with torch.no_grad():
            self.head.bias.copy_(torch.from_numpy(inverse.astype(np.float32)))

    def forward(self, x, future):
        if self.kind == 'mlp':
            h = self.encoder(x)
        else:
            _, state = self.encoder(x)
            h = (state[0] if self.kind == 'lstm' else state)[-1]
        h = torch.cat((self.dropout(h), future.flatten(1)), dim=1)
        return nn.functional.softplus(self.head(h)).reshape(-1, 24, self.regions, 2)


def tensor_inputs(bundle, data):
    z = (bundle['x'] - data.region_mean) / data.region_std
    mature = (np.arange(-168, 0) + 1 + 24 <= 0).astype(np.float32)
    maturity = np.broadcast_to(mature[None, :, None], (len(z), 168, 1))
    x = np.concatenate((z.reshape(len(z), 168, -1), bundle['past'], maturity), axis=-1)
    return (torch.from_numpy(x.copy()), torch.from_numpy(bundle['future']),
            torch.from_numpy((bundle['y'] / data.channel_std).astype(np.float32)))


def ridge_features(bundle, data, region):
    z = (bundle['x'][:, :, region] - data.region_mean[region]) / data.region_std[region]
    q = bundle['origins']
    # Direct 24-step AR with full168h of both targets and origin calendar dummies.
    return np.concatenate((z.reshape(len(z), -1), np.eye(24)[data.hour[q]],
                           np.eye(7)[data.weekday[q]]), axis=1).astype(np.float64)


def fit_ridge(train, valid, data, alphas, progress):
    predictions = [np.empty_like(valid['y']) for _ in alphas]
    weights, offsets = [[] for _ in alphas], [[] for _ in alphas]
    for r in range(len(data.ids)):
        x = ridge_features(train, data, r)
        xv = ridge_features(valid, data, r)
        y = (train['y'][:, :, r] / data.channel_std).reshape(len(x), -1).astype(np.float64)
        xm, ym = x.mean(0), y.mean(0)
        xc, yc = x-xm, y-ym
        gram, rhs = xc.T @ xc, xc.T @ yc
        eye = np.eye(x.shape[1])
        for a, alpha in enumerate(alphas):
            coef = np.linalg.solve(gram + alpha*eye, rhs)
            bias = ym-xm@coef
            p = ((xv@coef+bias).reshape(len(xv), 24, 2)*data.channel_std)
            predictions[a][:, :, r] = np.maximum(p, 0)
            weights[a].append(coef)
            offsets[a].append(bias)
        if r % 16 == 0:
            progress({'stage': 'ridge_fit', 'regions_completed': r+1})
    states = [{'coef': np.stack(w), 'bias': np.stack(b)} for w, b in zip(weights, offsets)]
    return predictions, states


def predict_ridge(bundle, data, state):
    p = np.empty_like(bundle['y'])
    for r in range(len(data.ids)):
        x = ridge_features(bundle, data, r)
        p[:, :, r] = np.maximum((x@state['coef'][r]+state['bias'][r]).reshape(len(x), 24, 2)
                               * data.channel_std, 0)
    return p
