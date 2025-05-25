import torch
from torchvision import transforms

from data.graph_dataset import GraphDataset
from models.extractor import GraphExtractor
from trainer import Trainer
from data.preprocess import preprocess_image
from data.metrics import adjacency_matrix_accuracy

if __name__ == "__main__":
    csv_path = "/home/Utkarsh/pCLUB/data/graphs.csv"
    image_folder = "/home/Utkarsh/pCLUB/data/images"

    # Initialize dataset
    dataset = GraphDataset(csv_path, image_folder, max_nodes=10, transform=preprocess_image)

    # Initialize model
    model = GraphExtractor()

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        dataset=dataset,
        preprocess_fn=preprocess_image,
        metric_fn=adjacency_matrix_accuracy,
        batch_size=16,
        epochs=30,
        lr=1e-3,
        weight_decay=0.0
    )

    trainer.train()

    # ✅ Save trained model
    torch.save(model.state_dict(), "model.pt")
    print("✅ model.pt saved.")
