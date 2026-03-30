# model.py
# Flood risk models:
#   1. GCN        -- simple baseline
#   2. QGNN       -- hybrid quantum-classical GNN (main contribution)
#
# TODO: tune hidden dims
# TODO: figure out how to properly call the qnode from inside nn.Module
#       QuantumLayer.forward() is currently a stub (returns zeros)
# TODO: try more qubits / deeper VQC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pennylane as qml

# ---------------------------------------------------------------------------
# 1. GCN baseline
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """
    Simple 3-layer GCN for node-level binary classification.
    Input features: 4 (will be 6 once slope/dist features are added to data.py)
    """

    def __init__(self, in_channels=4, hidden=64, out_channels=1):
        super().__init__()
        # TODO: try adding batch norm
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, 32)
        self.head  = nn.Linear(32, out_channels)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.drop(F.relu(self.conv1(x, edge_index)))
        x = self.drop(F.relu(self.conv2(x, edge_index)))
        x = F.relu(self.conv3(x, edge_index))
        return self.head(x)   # logits (N, 1)


# ---------------------------------------------------------------------------
# 2. Quantum layer
# ---------------------------------------------------------------------------

N_QUBITS = 4

# NOTE: using default.qubit because lightning.qubit does not support
#       parameter broadcasting (needed to process all nodes at once)
dev = qml.device('default.qubit', wires=N_QUBITS)


@qml.qnode(dev, interface='torch', diff_method='backprop')
def vqc(inputs, weights):
    """
    4-qubit variational circuit.
      inputs  : (B, 4)  -- angle encoding per qubit
      weights : (n_layers, 4, 3)  -- Rot gate angles

    TODO: experiment with more expressive ansatz
    TODO: try IQP encoding or amplitude encoding
    """
    # Angle encoding
    for q in range(N_QUBITS):
        qml.RY(inputs[:, q], wires=q)

    # Variational layers: Rot + CNOT ring
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        for q in range(N_QUBITS):
            qml.Rot(weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    # Measure all qubits
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


class QuantumLayer(nn.Module):
    """
    Wraps the VQC as a standard nn.Module.
    Input : (B, 32)  classical features
    Output: (B, 4)   quantum measurements
    """

    def __init__(self, n_layers=2):
        super().__init__()
        self.encode  = nn.Sequential(nn.Linear(32, N_QUBITS), nn.Tanh())
        # Store weights as nn.Parameter (TorchLayer broken in PennyLane 0.44)
        self.weights = nn.Parameter(
            torch.randn(n_layers, N_QUBITS, 3) * 0.1
        )

    def forward(self, x):
        # TODO: actually call the VQC here
        # not sure how to pass self.weights into the qnode correctly yet
        # returning zeros as a placeholder so the rest of the model at least runs
        B = x.shape[0]
        return torch.zeros(B, N_QUBITS)


# ---------------------------------------------------------------------------
# 3. QGNN  (hybrid quantum-classical)
# ---------------------------------------------------------------------------

class QGNN(nn.Module):
    """
    Hybrid model:
      Classical GCN encoder -> Quantum layer -> skip connection -> GCN -> MLP head

    TODO: try deeper classical encoder
    TODO: decide if skip connection is worth the extra params
    """

    def __init__(self, in_channels=4, out_channels=1, n_q_layers=2):
        super().__init__()
        # Classical encoder
        self.enc1 = GCNConv(in_channels, 32)
        self.enc2 = GCNConv(32, 32)

        # Quantum layer
        self.quantum = QuantumLayer(n_layers=n_q_layers)

        # After cat([classical(32), quantum(4)]) = 36 channels
        self.dec  = GCNConv(36, 32)
        self.head = nn.Linear(32, out_channels)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        h = self.drop(F.relu(self.enc1(x, edge_index)))
        h = F.relu(self.enc2(h, edge_index))     # (N, 32)

        q = self.quantum(h)                      # (N, 4)

        combined = torch.cat([h, q], dim=1)      # (N, 36)
        combined = self.drop(F.relu(self.dec(combined, edge_index)))
        return self.head(combined)               # logits (N, 1)


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    x  = torch.randn(10, 4)
    ei = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)

    gcn  = GCN(in_channels=4)
    qgnn = QGNN(in_channels=4)

    print('GCN  params:', count_params(gcn))
    print('QGNN params:', count_params(qgnn))

    out_gcn  = gcn(x, ei)
    out_qgnn = qgnn(x, ei)

    print('GCN  output shape:', out_gcn.shape)
    print('QGNN output shape:', out_qgnn.shape)
    print('model.py OK')
