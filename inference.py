import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

from models.extractor import GraphExtractor
from data.preprocess import preprocess_image
from fuel_solver import compute_min_fuel

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess_image(image)  # returns (1, 128, 128)
    return image.unsqueeze(0)        # shape (1, 1, 128, 128)

def visualize_adjacency(adj_matrix, out_path="visuals/infer_result.png"):
    os.makedirs("visuals", exist_ok=True)
    plt.imshow(adj_matrix, cmap="viridis")
    plt.title("Predicted Adjacency")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(out_path)
    print(f"[✓] Saved visualization to: {out_path}")
    plt.close()

def run_inference():
    image_path = "data/images/2.png"
    model_path = "model.pt"
    max_nodes = 10
    threshold = 0.3

    print(f"📥 Loading image: {image_path}")
    print(f"📦 Loading model: {model_path}")

    # ✅ Load GraphExtractor to match saved weights
    model = GraphExtractor()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Preprocess image
    image_tensor = load_image(image_path)

    # Predict adjacency
    with torch.no_grad():
        output = model(image_tensor)[0]
        adj = output.detach().cpu().numpy().reshape(max_nodes, max_nodes)

    visualize_adjacency(adj > threshold)

    # Compute fuel
    fuel = compute_min_fuel(adj, threshold=threshold)
    print(f"🛢️  Minimum fuel required to reach Base {max_nodes - 1}: {fuel}")

if __name__ == "__main__":
    run_inference()
