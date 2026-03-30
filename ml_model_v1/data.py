# data.py
# Building a dataset for flood risk prediction on Indian river basins.
# Using ERA5-Land rainfall + SRTM elevation data.
#
# TODO: double-check the lat/lon bounding box for Ganga-Brahmaputra
# TODO: figure out a good flood threshold
# TODO: add slope + dist-to-river features later

import os
import numpy as np
import requests
import torch
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import NearestNeighbors

# Basin: rough grid over northern India
# 5 lat points x 10 lon points = 50 nodes
LATS = np.linspace(24.0, 29.0, 5)
LONS = np.linspace(76.0, 92.0, 10)

START = '2018-01-01'
END   = '2020-09-28'

OPENMETEO_URL = 'https://archive-api.open-meteo.com/v1/archive'


def fetch_rainfall(lats, lons):
    """Download daily precipitation for each grid point from ERA5-Land."""
    # TODO: add retry logic
    # TODO: cache results so we don't re-download every run

    all_series = []
    n = len(lats)

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        print(f'  fetching station {i+1}/{n}  lat={lat:.1f} lon={lon:.1f}')
        resp = requests.get(OPENMETEO_URL, params={
            'latitude'  : float(lat),
            'longitude' : float(lon),
            'start_date': START,
            'end_date'  : END,
            'daily'     : 'precipitation_sum',
            'timezone'  : 'Asia/Kolkata',
        }, timeout=30)
        resp.raise_for_status()
        series = resp.json()['daily']['precipitation_sum']
        # replace None with 0.0
        series = [v if v is not None else 0.0 for v in series]
        all_series.append(series)

    rainfall = np.array(all_series, dtype=np.float32).T  # (T, N)
    print(f'rainfall shape: {rainfall.shape}')
    return rainfall


def fetch_elevation(lats, lons):
    """Fetch SRTM elevation (metres) for each grid point."""
    import srtm
    sdata = srtm.get_data()
    elev = np.zeros(len(lats), dtype=np.float32)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        val = sdata.get_elevation(float(lat), float(lon))
        elev[i] = val if val is not None else 100.0  # fallback: plains
    return elev


def build_edges(xy, k=5):
    """kNN spatial edges (undirected)."""
    # TODO: also add downstream flow edges based on elevation gradient
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nbrs.kneighbors(xy)
    src, dst = [], []
    for i in range(len(xy)):
        for j in idx[i, 1:]:   # skip self (idx[i,0] == i)
            src += [i, int(j)]
            dst += [int(j), i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def make_labels(rainfall, elev, window=3, threshold=0.55):
    """
    Simple flood risk score per node per day.
    Score combines recent rainfall and low elevation.
    If score > threshold -> flood = 1.

    TODO: this is pretty naive, refine the scoring formula
    TODO: consider upstream accumulation
    """
    T, N = rainfall.shape
    rain_max = rainfall.max() + 1e-6
    elev_n   = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)

    labels = np.zeros((T - window, N), dtype=np.float32)
    for t in range(window, T):
        rain_w = rainfall[t - window:t].mean(axis=0) / rain_max
        score  = 0.6 * rain_w + 0.4 * (1 - elev_n)
        labels[t - window] = (score > threshold).astype(np.float32)

    print(f'flood rate: {labels.mean():.3f}')
    return labels


class FloodDataset(Dataset):
    """
    One PyG graph per timestep.
    Node features: [rain_t-3, rain_t-2, rain_t-1, elev_norm]
    (4 features for now -- will add slope & dist_river later)
    """

    def __init__(self, root='data', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['basin.npz']

    @property
    def processed_file_names(self):
        # TODO: don't hardcode 999 -- read from npz
        return [f'graph_{i:04d}.pt' for i in range(999)]

    def download(self):
        lon_grid, lat_grid = np.meshgrid(LONS, LATS)
        lats = lat_grid.flatten().astype(np.float32)
        lons = lon_grid.flatten().astype(np.float32)

        # rough km projection
        x = (lons - lons.mean()) * 111.0 * np.cos(np.radians(lats.mean()))
        y = (lats - lats.mean()) * 111.0
        xy = np.stack([x, y], axis=1)

        rainfall = fetch_rainfall(lats, lons)
        elev     = fetch_elevation(lats, lons)
        edge_idx = build_edges(xy)
        labels   = make_labels(rainfall, elev)

        np.savez(os.path.join(self.raw_dir, 'basin.npz'),
                 xy=xy, rainfall=rainfall, elev=elev,
                 labels=labels, edge_index=edge_idx.numpy())
        print('[download] done')

    def process(self):
        data = np.load(os.path.join(self.raw_dir, 'basin.npz'))
        rainfall   = data['rainfall']                           # (T, N)
        elev       = data['elev']                               # (N,)
        labels     = data['labels']                             # (T-3, N)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long)

        T       = rainfall.shape[0]
        window  = 3
        rain_mx = rainfall.max() + 1e-6
        elev_n  = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)

        for i in range(T - window):
            t     = i + window
            rain3 = rainfall[t-3:t] / rain_mx   # (3, N)
            x = torch.tensor(np.stack([
                rain3[0], rain3[1], rain3[2], elev_n
            ], axis=1), dtype=torch.float32)     # (N, 4)

            y = torch.tensor(labels[i], dtype=torch.float32).unsqueeze(1)
            g = Data(x=x, edge_index=edge_index, y=y)
            torch.save(g, os.path.join(self.processed_dir, f'graph_{i:04d}.pt'))

        print(f'[process] saved {T - window} graphs')

    def len(self):
        # TODO: read from metadata instead of hardcoding
        return 999

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'graph_{idx:04d}.pt'),
            weights_only=False,
        )


if __name__ == '__main__':
    ds = FloodDataset(root='data')
    g  = ds[0]
    print('sample graph:', g)
    print('x shape:', g.x.shape)
    print('y shape:', g.y.shape)
