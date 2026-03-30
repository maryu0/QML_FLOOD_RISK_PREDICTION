"""
Synthetic spatio-temporal flood-risk dataset for PyTorch Geometric.

Dataset design
--------------
- Nodes: 50 gauges in a river basin
- Node features: [rain_t-3, rain_t-2, rain_t-1, elevation, slope, dist_to_river]
- Edges: undirected kNN spatial edges + directed downstream flow edges
- Labels: node-wise binary flood risk at time t
- Time steps: 1000 by default

This module creates realistic synthetic correlations:
1. Rainfall fields are spatially and temporally correlated.
2. Upstream rainfall contributes to downstream discharge pressure.
3. Low elevation and river proximity increase flood susceptibility.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset


@dataclass
class FloodDataConfig:
    n_nodes: int = 50
    n_timesteps: int = 1000
    lookback: int = 3
    k_neighbors: int = 5
    flood_quantile: float = 0.72
    seed: int = 42


def _make_node_geometry(n_nodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic station positions and static geo features."""
    rng = np.random.default_rng(seed)

    # Create clustered 2D positions to mimic sub-basins.
    centers = np.array([[0.2, 0.3], [0.7, 0.4], [0.5, 0.8]], dtype=np.float32)
    assignments = rng.integers(0, len(centers), size=n_nodes)
    pos = centers[assignments] + rng.normal(0.0, 0.09, size=(n_nodes, 2)).astype(np.float32)
    pos = np.clip(pos, 0.0, 1.0)

    # Elevation gently decreases from north-west to south-east + terrain noise.
    elevation = 1.0 - (0.65 * pos[:, 0] + 0.35 * pos[:, 1])
    elevation += rng.normal(0.0, 0.04, size=n_nodes)
    elevation = np.clip(elevation, 0.0, 1.0)

    # Slope correlates with local terrain roughness.
    slope = 0.35 * elevation + 0.65 * rng.random(n_nodes)
    slope = np.clip(slope, 0.0, 1.0)

    # River corridor centered diagonally; low distance means river-adjacent.
    river_line = 0.6 * pos[:, 0] + 0.4 * pos[:, 1]
    dist_to_river = np.abs(river_line - np.median(river_line))
    dist_to_river = dist_to_river / (dist_to_river.max() + 1e-8)

    static = np.stack([elevation, slope, dist_to_river], axis=1).astype(np.float32)
    return pos.astype(np.float32), static


def _build_edge_index(pos: np.ndarray, elevation: np.ndarray, k_neighbors: int) -> torch.Tensor:
    """Build kNN undirected edges plus directed downstream edges."""
    n_nodes = pos.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(pos)
    _, knn_idx = nbrs.kneighbors(pos)

    src_list: list[int] = []
    dst_list: list[int] = []

    # Undirected kNN edges.
    for i in range(n_nodes):
        for j in knn_idx[i, 1:]:
            src_list.extend([i, int(j)])
            dst_list.extend([int(j), i])

    # Directed downhill flow edges: i -> nearest lower elevation neighbor.
    nbrs_flow = NearestNeighbors(n_neighbors=min(10, n_nodes)).fit(pos)
    _, flow_idx = nbrs_flow.kneighbors(pos)
    for i in range(n_nodes):
        lower = [int(j) for j in flow_idx[i, 1:] if elevation[int(j)] < elevation[i]]
        if not lower:
            continue
        best = min(lower, key=lambda j: float(np.linalg.norm(pos[i] - pos[j])))
        src_list.append(i)
        dst_list.append(best)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def _upstream_map(edge_index: torch.Tensor, n_nodes: int) -> list[list[int]]:
    """Return upstream neighbor list for each node using directed edges as pressure paths."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    upstream: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, d in zip(src, dst):
        upstream[int(d)].append(int(s))
    return upstream


def _generate_rainfall(n_timesteps: int, n_nodes: int, pos: np.ndarray, seed: int) -> np.ndarray:
    """Generate spatially and temporally correlated rainfall."""
    rng = np.random.default_rng(seed)

    phase = rng.uniform(0, 2 * math.pi, size=n_nodes)
    local_amp = 0.4 + 0.6 * rng.random(n_nodes)

    rain = np.zeros((n_timesteps, n_nodes), dtype=np.float32)
    rain_prev = 0.2 * rng.random(n_nodes)

    spatial_wave = 0.5 * pos[:, 0] + 0.5 * pos[:, 1]

    for t in range(n_timesteps):
        seasonal = 0.55 + 0.45 * math.sin(2 * math.pi * (t / 180.0))
        monsoon_spike = 1.0 if (t % 365) in range(130, 260) else 0.0
        storm_global = rng.gamma(shape=1.7 + 2.0 * monsoon_spike, scale=0.45)

        node_pattern = np.sin(2 * math.pi * (t / 24.0) + phase) * local_amp
        node_pattern = 0.5 + 0.5 * (node_pattern - node_pattern.min()) / (np.ptp(node_pattern) + 1e-8)

        noise = rng.normal(0.0, 0.08, size=n_nodes)
        rain_t = (
            0.55 * rain_prev
            + 0.30 * seasonal * node_pattern
            + 0.35 * storm_global
            + 0.18 * spatial_wave
            + noise
        )

        rain_t = np.clip(rain_t, 0.0, 3.0)
        rain[t] = rain_t.astype(np.float32)
        rain_prev = rain_t

    return rain


def _generate_labels(
    rainfall: np.ndarray,
    static: np.ndarray,
    edge_index: torch.Tensor,
    lookback: int,
    flood_quantile: float,
) -> np.ndarray:
    """Compute binary flood labels with upstream propagation."""
    t_total, n_nodes = rainfall.shape
    elevation = static[:, 0]
    slope = static[:, 1]
    dist_river = static[:, 2]

    elev_inv = 1.0 - elevation
    slope_risk = slope
    river_risk = 1.0 - dist_river

    upstream = _upstream_map(edge_index, n_nodes)

    scores = np.zeros((t_total - lookback, n_nodes), dtype=np.float32)

    rain_scale = np.percentile(rainfall, 90) + 1e-8

    for t in range(lookback, t_total):
        rain_hist = rainfall[t - lookback:t]
        rain_avg = rain_hist.mean(axis=0) / rain_scale
        rain_avg = np.clip(rain_avg, 0.0, 2.0)

        upstream_pressure = np.zeros(n_nodes, dtype=np.float32)
        for node in range(n_nodes):
            ups = upstream[node]
            if ups:
                upstream_pressure[node] = float(np.mean(rain_avg[np.array(ups)]))

        score = (
            0.44 * rain_avg
            + 0.20 * elev_inv
            + 0.12 * slope_risk
            + 0.12 * river_risk
            + 0.12 * upstream_pressure
        )
        scores[t - lookback] = score

    threshold = float(np.quantile(scores, flood_quantile))
    labels = (scores >= threshold).astype(np.float32)
    return labels


def generate_synthetic_flood_graphs(config: FloodDataConfig) -> tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Generate all primitives for graph time-series construction.

    Returns
    -------
    pos: (N, 2)
    static: (N, 3)
    edge_index: (2, E)
    rainfall: (T, N)
    labels: (T-lookback, N)
    """
    pos, static = _make_node_geometry(config.n_nodes, config.seed)
    edge_index = _build_edge_index(pos, static[:, 0], config.k_neighbors)
    rainfall = _generate_rainfall(config.n_timesteps, config.n_nodes, pos, config.seed)
    labels = _generate_labels(
        rainfall,
        static,
        edge_index,
        config.lookback,
        config.flood_quantile,
    )
    return pos, static, edge_index, rainfall, labels


class IndianFloodDataset(Dataset):
    """Synthetic Indian basin flood dataset as PyG graph snapshots."""

    def __init__(
        self,
        root: str = "data",
        n_nodes: int = 50,
        n_timesteps: int = 1000,
        lookback: int = 3,
        k_neighbors: int = 5,
        flood_quantile: float = 0.72,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.config = FloodDataConfig(
            n_nodes=n_nodes,
            n_timesteps=n_timesteps,
            lookback=lookback,
            k_neighbors=k_neighbors,
            flood_quantile=flood_quantile,
            seed=seed,
        )
        self.dataset_tag = (
            f"synthetic_n{n_nodes}_t{n_timesteps}_"
            f"lb{lookback}_k{k_neighbors}_s{seed}"
        )
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw", self.dataset_tag)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", self.dataset_tag)

    @property
    def raw_file_names(self) -> list[str]:
        return ["synthetic_flood_raw.npz"]

    @property
    def processed_file_names(self) -> list[str]:
        n_graphs = self.config.n_timesteps - self.config.lookback
        return [f"graph_{i:04d}.pt" for i in range(n_graphs)]

    def download(self) -> None:
        pos, static, edge_index, rainfall, labels = generate_synthetic_flood_graphs(self.config)

        np.savez(
            os.path.join(self.raw_dir, "synthetic_flood_raw.npz"),
            pos=pos,
            static=static,
            rainfall=rainfall,
            labels=labels,
            edge_index=edge_index.cpu().numpy(),
            config=np.array([
                self.config.n_nodes,
                self.config.n_timesteps,
                self.config.lookback,
                self.config.k_neighbors,
                self.config.flood_quantile,
                self.config.seed,
            ], dtype=np.float32),
        )

    def process(self) -> None:
        raw = np.load(os.path.join(self.raw_dir, "synthetic_flood_raw.npz"))
        static = raw["static"]
        rainfall = raw["rainfall"]
        labels = raw["labels"]
        edge_index = torch.tensor(raw["edge_index"], dtype=torch.long)

        lookback = self.config.lookback
        n_graphs = rainfall.shape[0] - lookback

        rain_scale = float(np.percentile(rainfall, 95) + 1e-8)

        elev = static[:, 0]
        slope = static[:, 1]
        dist = static[:, 2]

        elev_n = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)
        slope_n = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)

        elev_t = torch.tensor(elev_n, dtype=torch.float32).unsqueeze(1)
        slope_t = torch.tensor(slope_n, dtype=torch.float32).unsqueeze(1)
        dist_t = torch.tensor(dist, dtype=torch.float32).unsqueeze(1)

        for i in range(n_graphs):
            t = i + lookback
            rain_window = rainfall[t - lookback:t] / rain_scale
            rain_cols = [
                torch.tensor(rain_window[j], dtype=torch.float32).unsqueeze(1)
                for j in range(lookback)
            ]
            x = torch.cat(rain_cols + [elev_t, slope_t, dist_t], dim=1)
            y = torch.tensor(labels[i], dtype=torch.float32).unsqueeze(1)

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                num_nodes=self.config.n_nodes,
            )
            torch.save(graph, os.path.join(self.processed_dir, f"graph_{i:04d}.pt"))

    def len(self) -> int:
        return self.config.n_timesteps - self.config.lookback

    def get(self, idx: int) -> Data:
        return torch.load(
            os.path.join(self.processed_dir, f"graph_{idx:04d}.pt"),
            weights_only=False,
        )


def temporal_split(
    dataset: IndianFloodDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Chronological split to avoid temporal leakage."""
    n = len(dataset)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    idx = list(range(n))
    return dataset[idx[:train_end]], dataset[idx[train_end:val_end]], dataset[idx[val_end:]]


def dataset_stats(dataset: IndianFloodDataset) -> dict[str, float]:
    """Quick summary for experiment logs and paper tables."""
    sample = dataset[0]
    n_check = min(200, len(dataset))
    flood_rates = [float(dataset[i].y.mean().item()) for i in range(n_check)]
    return {
        "n_graphs": float(len(dataset)),
        "n_nodes": float(sample.num_nodes),
        "n_features": float(sample.x.shape[1]),
        "n_edges": float(sample.edge_index.shape[1]),
        "flood_rate": float(np.mean(flood_rates)),
        "x_min": float(sample.x.min().item()),
        "x_max": float(sample.x.max().item()),
    }


if __name__ == "__main__":
    ds = IndianFloodDataset(root="data", n_timesteps=1000, n_nodes=50)
    stats = dataset_stats(ds)
    print("Synthetic IndianFloodDataset ready")
    for key, value in stats.items():
        print(f"{key}: {value}")
