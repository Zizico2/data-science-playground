import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Dict, List, Optional, Final

# Type Aliases for clarity in GNN contexts
NodeFeatures = Tensor  # Shape: [N, Data_Dim]
EdgeIndex = Tensor  # Shape: [2, E]
EdgeAttr = Tensor  # Shape: [E, Edge_Dim]
Targets = Tensor  # Shape: [N, 1]


class HVIDataGenerator:
    """
    Generates synthetic socioeconomic data for GNN-based Housing Velocity analysis.
    Designed for testing high-head Attention mechanisms and Structure Learning.
    """

    def __init__(
        self, num_nodes: int = 5000, num_features: int = 32, seed: int = 42
    ) -> None:
        self.num_nodes: Final[int] = num_nodes
        self.num_features: Final[int] = num_features
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate_node_data(self) -> Tuple[NodeFeatures, Targets]:
        """
        Creates a synthetic feature matrix with built-in socioeconomic correlations.

        Returns:
            Tuple containing:
            - Normalized Node Features [N, 32]
            - HVI Target Variables [N, 1]
        """
        # 1. Latent Prosperity Factor: The underlying driver of correlations
        prosperity: Tensor = torch.randn(self.num_nodes)

        # 2. Economy & Labor Features
        # Median Income: correlated with prosperity
        median_income: Tensor = (
            55000 + (prosperity * 15000) + (torch.randn(self.num_nodes) * 2000)
        )

        # STEM Industry %: High in prosperous nodes
        stem_pct: Tensor = torch.sigmoid(prosperity - 1.0) * 0.35

        # Remote Work %: The "Gentrification Trigger"
        remote_work_pct: Tensor = torch.sigmoid(prosperity * 1.2) * 0.45

        # 3. Demographic Features
        bach_degree_pct: Tensor = (
            torch.sigmoid(prosperity + torch.randn(self.num_nodes) * 0.4) * 0.55
        )
        median_age: Tensor = 38 + (torch.randn(self.num_nodes) * 6)

        # 4. Housing Metrics
        # House Prices: Driven by local wages and desirability (prosperity)
        median_price: Tensor = (
            250000 + (prosperity * 140000) + (bach_degree_pct * 90000)
        )
        vacancy_rate: Tensor = torch.clamp(
            0.12 - (prosperity * 0.04), min=0.01, max=0.25
        )

        # 5. Calculate Synthetic HVI (The Ground Truth for Training)
        # HVI logic: High where remote work is high but wages are not yet peak (The 'Lisbon' effect)
        annual_appreciation: Tensor = (
            (prosperity * 18000)
            + (remote_work_pct * 45000)
            + (torch.randn(self.num_nodes) * 4000)
        )
        hvi_labels: Targets = (annual_appreciation / median_income).unsqueeze(1)

        # 6. Assemble Feature Matrix
        x: NodeFeatures = torch.zeros((self.num_nodes, self.num_features))

        # Mapping indices to features for XAI reference
        x[:, 0] = median_income
        x[:, 1] = stem_pct
        x[:, 2] = remote_work_pct
        x[:, 3] = bach_degree_pct
        x[:, 4] = median_age
        x[:, 5] = median_price
        x[:, 6] = vacancy_rate

        # Fill noise/auxiliary features [7:32]
        x[:, 7:] = torch.randn((self.num_nodes, self.num_features - 7))

        # 7. Z-Score Normalization
        x_norm: NodeFeatures = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

        return x_norm.to(self.device), hvi_labels.to(self.device)

    def generate_edge_attributes(
        self, edge_index: EdgeIndex, x_raw: NodeFeatures
    ) -> EdgeAttr:
        """
        Calculates Edge Deltas (Arbitrage Signals) for the GNN.

        Args:
            edge_index: The connectivity matrix [2, E]
            x_raw: Raw (unnormalized) features to calculate meaningful deltas.

        Returns:
            Tensor of edge features [E, Edge_Dim]
        """
        src, dst = edge_index[0], edge_index[1]

        # Difference in Income (Income_source - Income_target)
        # If Positive: Capital is flowing from a high-wage area to a low-wage area
        income_delta: Tensor = x_raw[src, 0] - x_raw[dst, 0]

        # Price Gap (Price_source / Price_target)
        # Ratio > 1.0 implies the target is "cheap" relative to the source
        price_ratio: Tensor = x_raw[src, 5] / (x_raw[dst, 5] + 1e-6)

        # STEM Similarity (Cosine similarity of industry profiles)
        # We use a simple absolute difference for the synthetic example
        stem_diff: Tensor = torch.abs(x_raw[src, 1] - x_raw[dst, 1])

        edge_attr: EdgeAttr = torch.stack([income_delta, price_ratio, stem_diff], dim=1)

        # Normalize edge attributes
        edge_attr_norm: EdgeAttr = (edge_attr - edge_attr.mean(dim=0)) / (
            edge_attr.std(dim=0) + 1e-6
        )

        return edge_attr_norm.to(self.device)
