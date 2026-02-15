import torch
from data_science_playground.test import HVIDataGenerator, EdgeIndex, EdgeAttr


def main() -> None:
    # Initialize Generator
    gen: HVIDataGenerator = HVIDataGenerator(num_nodes=3000)

    # Generate Node Data
    nodes, targets = gen.generate_node_data()

    # Create a Dummy Edge Index (e.g., random connections)
    # In reality, this would come from KNN or Structure Learning
    dummy_edges: EdgeIndex = torch.randint(0, 3000, (2, 15000))

    # Generate Edge Attributes (The "Deltas")
    edges: EdgeAttr = gen.generate_edge_attributes(dummy_edges, nodes)

    print(f"Nodes Tensor Type: {type(nodes)} | Shape: {nodes.shape}")
    print(f"Edges Tensor Type: {type(edges)} | Shape: {edges.shape}")
    print(f"Target Type: {type(targets)} | Mean HVI: {targets.mean().item():.4f}")
