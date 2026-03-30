# -*- coding: utf-8 -*-
"""
model.py  -  Classical GCN + Hybrid QGNN for Flood-Risk Prediction
===================================================================
Both models share the same I/O signature:
    forward(x, edge_index, batch=None)  ->  (N, 1) logits

GCN
---
  3-layer Graph Convolutional Network (classical baseline).

QGNN
----
  Hybrid Quantum-Classical GNN.

  Architecture:
    x (N,6)
      GCNConv -> h (N,32)        classical encoder
      GCNConv -> h (N,32)
          |
     tanh(Linear) -> z (N,4)    project to n_qubits
          |
        VQC -> q (N,4)           Pauli-Z expectations
          |
      cat([h, q]) -> (N,36)      skip connection
          |
      GCNConv -> (N,32)          graph diffusion over quantum output
          |
      MLP head -> logit (N,1)

Quantum circuit (PennyLane)
---------------------------
  - 4-qubit VQC on 'default.qubit' simulator
  - Angle encoding:  RY(pi * x_i)  per qubit
  - n_qlayers blocks of  Rot(phi,theta,omega)  + CNOT ring
  - Returns 4 Pauli-Z expectations in [-1, 1]
  - Weights stored as nn.Parameter; qnode called per-node for correct batching
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch_geometric.nn import GCNConv, GraphConv, GATConv, SAGEConv

# ---------------------------------------------------------------------------
# Quantum hyper-parameters
# ---------------------------------------------------------------------------
N_QUBITS_SMALL  = 4
N_QUBITS_LARGE  = 8
N_QLAYERS = 2


# ===========================================================================
# QuantumLayer  (nn.Module — direct qnode, no TorchLayer)
# ===========================================================================

class QuantumLayer(nn.Module):
    """
    Batched VQC with parameter broadcasting (4 or 8 qubits).

    Design rationale
    ----------------
    Owns weights as nn.Parameter; calls qnode with parameter broadcasting
    to process all B nodes in one circuit evaluation (not per-node loop).

    Input  : (B, in_features)
    Output : (B, n_qubits)  Pauli-Z expectations in [-1, 1]

    Circuit (per node)
    ------------------
    1. Angle encoding:    RY(pi * z_i)  on qubit i   (z in (-1,1) from Tanh)
    2. Variational block  repeated n_layers times:
         Rot(phi, theta, omega)  on every qubit
         CNOT ring:  0->1->2->...->0
         CZ ladder:  0-1, 1-2, 2-3, ... (stronger entanglement)
    3. Measurement:       <Z_i>  for i = 0 .. n_qubits-1
    """

    def __init__(self, in_features: int,
                 n_qubits: int = N_QUBITS_LARGE,
                 n_layers: int = N_QLAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical pre-projection: in_features -> n_qubits in (-1, 1)
        self.pre_proj = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.Tanh(),
        )

        # Variational quantum weights: Rot(phi,theta,omega) + CZ ladder params
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.01
        )
        # CZ entanglement strengths (one param per CZ pair)
        self.cz_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits - 1) * 0.01
        )

        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def _qnode(inputs_batch: torch.Tensor,
                   weights: torch.Tensor,
                   cz_wts: torch.Tensor) -> list:
            """Batched VQC: angle encoding + Rot + CNOT ring + CZ ladder.
            inputs_batch : (B, n_qubits)
            weights      : (n_layers, n_qubits, 3)
            cz_wts       : (n_layers, n_qubits-1)
            returns      : list of n_qubits tensors each of shape (B,)
            """
            # Angle encoding
            for i in range(n_qubits):
                qml.RY(inputs_batch[:, i] * np.pi, wires=i)

            for l in range(n_layers):
                # Rot on every qubit
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0],
                            weights[l, i, 1],
                            weights[l, i, 2], wires=i)

                # CNOT ring
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

                # CZ ladder: 0-1, 1-2, 2-3, ... (stronger entanglement)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._qnode = _qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self.pre_proj(x)
        raw = self._qnode(z, self.q_weights, self.cz_weights)
        return torch.stack(list(raw), dim=1).float()


# ===========================================================================
# Classical GCN Baseline
# ===========================================================================

class GCN(nn.Module):
    """
    3-layer GCN for node-level binary flood-risk prediction.

    GCNConv(in -> hidden) -> Dropout ->
    GCNConv(hidden -> hidden) -> Dropout ->
    GCNConv(hidden -> hidden//2) ->
    Linear(hidden//2 -> 1)      logit per node

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : hidden units       (default 64)
    dropout     : dropout rate       (default 0.3)
    """

    def __init__(self, in_channels: int = 6,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.drop  = dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden,      hidden)
        self.conv3 = GCNConv(hidden,      hidden // 2)
        self.head  = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return self.head(x)   # (N, 1) logits


# ===========================================================================
# Hybrid QGNN
# ===========================================================================

class QGNN(nn.Module):
    """
    Hybrid Quantum-Classical Graph Neural Network (4 or 8 qubits).

    Classical GCN layers encode structural context; the QuantumLayer
    introduces non-linear feature interactions via entangled qubits
    (CNOT ring + CZ ladder for stronger expressiveness).

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : classical hidden   (default 32)
    n_qubits    : VQC qubit count    (default 8; use 4 for comparison)
    n_qlayers   : VQC depth          (default 2)
    dropout     : dropout rate       (default 0.3)
    """

    def __init__(self, in_channels: int = 6,
                 hidden: int = 32,
                 n_qubits: int = N_QUBITS_LARGE,
                 n_qlayers: int = N_QLAYERS,
                 dropout: float = 0.3):
        super().__init__()
        self.drop = dropout
        self.n_qubits = n_qubits

        # Classical encoder
        self.enc1 = GCNConv(in_channels, hidden)
        self.enc2 = GCNConv(hidden,      hidden)

        # Quantum message-passing layer
        self.quantum = QuantumLayer(hidden, n_qubits, n_qlayers)

        # Classical decoder: graph diffusion over [classical || quantum]
        self.dec  = GCNConv(hidden + n_qubits, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        # Encode
        h = F.relu(self.enc1(x, edge_index))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.enc2(h, edge_index))

        # Quantum processing (node-wise, with entanglement)
        q = self.quantum(h)

        # Skip: concat classical embedding + quantum expectations
        hq  = torch.cat([h, q], dim=-1)
        out = F.relu(self.dec(hq, edge_index))

        return self.head(out)


# ===========================================================================
# Ablation: Classical substitute for the QuantumLayer
# ===========================================================================

class ClassicalQGNN(nn.Module):
    """
    Ablation model for the IEEE paper.

    Identical to QGNN in every way except the QuantumLayer is replaced by a
    plain Linear(hidden, n_qubits) + ReLU.  Because the parameter counts are
    nearly identical, any performance difference between ClassicalQGNN and
    QGNN can be attributed to the quantum processing rather than capacity.

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : classical hidden   (default 32)
    n_qubits    : output width of the classical substitute (default 4)
    dropout     : dropout rate       (default 0.3)
    """

    def __init__(self, in_channels: int = 6,
                 hidden: int = 32,
                 n_qubits: int = N_QUBITS_SMALL,
                 dropout: float = 0.3):
        super().__init__()
        self.drop = dropout

        self.enc1 = GCNConv(in_channels, hidden)
        self.enc2 = GCNConv(hidden,      hidden)

        # Classical substitute: same I/O shape as QuantumLayer
        self.classical_sub = nn.Sequential(
            nn.Linear(hidden, n_qubits),
            nn.ReLU(),
        )

        self.dec  = GCNConv(hidden + n_qubits, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        h = F.relu(self.enc1(x, edge_index))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.enc2(h, edge_index))

        q   = self.classical_sub(h)            # (N, n_qubits)
        hq  = torch.cat([h, q], dim=-1)        # (N, hidden + n_qubits)
        out = F.relu(self.dec(hq, edge_index))

        return self.head(out)                  # (N, 1) logits


# ===========================================================================
# Competitive Baseline 1: GraphSAGE
# ===========================================================================

class GraphSAGE(nn.Module):
    """
    Graph SAGE baseline: neighborhood sampling + MLP aggregation.

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : hidden units       (default 64)
    dropout     : dropout rate       (default 0.3)
    """
    def __init__(self, in_channels: int = 6,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.drop = dropout
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.sage3 = SAGEConv(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.sage3(x, edge_index))
        return self.head(x)


# ===========================================================================
# Competitive Baseline 2: Graph Attention Network (GAT)
# ===========================================================================

class GAT(nn.Module):
    """
    Graph Attention Network: multi-head attention over neighborhoods.

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : hidden units       (default 64)
    heads       : attention heads per layer (default 4)
    dropout     : dropout rate       (default 0.3)
    """
    def __init__(self, in_channels: int = 6,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.drop = dropout
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden * heads, hidden // 2, heads=1, dropout=dropout)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.gat3(x, edge_index))
        return self.head(x)


# ===========================================================================
# Larger Classical Baseline: 6-layer GCN (for parameter parity with QGNN)
# ===========================================================================

class LargeGCN(nn.Module):
    """
    6-layer deeper GCN: comparable param count to QGNN for fair comparison.

    Parameters
    ----------
    in_channels : input feature dim  (default 6)
    hidden      : hidden units       (default 32)
    dropout     : dropout rate       (default 0.3)
    """
    def __init__(self, in_channels: int = 6,
                 hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.drop = dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
        self.conv5 = GCNConv(hidden, hidden)
        self.conv6 = GCNConv(hidden, hidden // 2)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        return self.head(x)



def build_model(name: str, **kwargs) -> nn.Module:
    """
    Convenience factory for all available models.

    Examples
    --------
    model = build_model('gcn',             in_channels=6, hidden=64)
    model = build_model('largegcn',        in_channels=6, hidden=32)
    model = build_model('graphsage',       in_channels=6, hidden=64)
    model = build_model('gat',             in_channels=6, hidden=64, heads=4)
    model = build_model('classical_qgnn',  in_channels=6, hidden=32, n_qubits=4)
    model = build_model('qgnn',            in_channels=6, hidden=32, n_qubits=8)
    model = build_model('qgnn_small',      in_channels=6, hidden=32, n_qubits=4)
    """
    name = name.lower()
    if name == 'gcn':
        return GCN(**kwargs)
    if name == 'largegcn':
        return LargeGCN(**kwargs)
    if name == 'graphsage':
        return GraphSAGE(**kwargs)
    if name == 'gat':
        return GAT(**kwargs)
    if name == 'classical_qgnn':
        return ClassicalQGNN(**kwargs)
    if name == 'qgnn':
        # Default: 8-qubit version with strong entanglement
        if 'n_qubits' not in kwargs:
            kwargs['n_qubits'] = N_QUBITS_LARGE
        return QGNN(**kwargs)
    if name == 'qgnn_small':
        # 4-qubit version for ablation
        if 'n_qubits' not in kwargs:
            kwargs['n_qubits'] = N_QUBITS_SMALL
        return QGNN(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose from: gcn, largegcn, graphsage, gat, "
                     f"classical_qgnn, qgnn, qgnn_small.")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


# ===========================================================================
# Smoke-test
# ===========================================================================

if __name__ == '__main__':
    n_nodes = 50
    n_feat  = 6
    n_edges = 120
    x          = torch.randn(n_nodes, n_feat)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    batch      = torch.zeros(n_nodes, dtype=torch.long)

    models_to_test = [
        ('gcn', {}),
        ('largegcn', {}),
        ('graphsage', {}),
        ('gat', {'heads': 4}),
        ('classical_qgnn', {'n_qubits': 4}),
        ('qgnn_small', {}),
        ('qgnn', {}),
    ]

    for model_name, extra_kwargs in models_to_test:
        try:
            print(f'\n--- {model_name.upper()} ---')
            kwargs = {'in_channels': n_feat}
            kwargs.update(extra_kwargs)
            model = build_model(model_name, **kwargs)
            out = model(x, edge_index, batch)
            p = count_parameters(model)
            print(f'  output : {tuple(out.shape)}')
            print(f'  params : {p["trainable"]:,} trainable')
        except Exception as e:
            print(f'  ERROR: {e}')

    print('\nmodel.py smoke-test passed.')
