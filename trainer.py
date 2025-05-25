from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import os
from fuel_solver import compute_min_fuel

class Trainer:
    def __init__(self, model, dataset, preprocess_fn, metric_fn, batch_size=16, epochs=30, lr=1e-3, weight_decay=0.0):
        self.model = model
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn
        self.metric_fn = metric_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        pos_weight = self.compute_pos_weight(self.dataset).to(self.device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            total_metric = 0
            count = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels.view(outputs.shape))
                sigmoid_outputs = torch.sigmoid(outputs)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                total_metric += self.metric_fn(sigmoid_outputs, labels.view(outputs.shape)).item()
                count += 1

            avg_loss = epoch_loss / count
            avg_metric = total_metric / count
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f} - Train Metric: {avg_metric:.4f}")

            self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_metric = 0
        count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels.view(outputs.shape))
                sigmoid_outputs = torch.sigmoid(outputs)
                val_loss += loss.item()
                val_metric += self.metric_fn(sigmoid_outputs, labels.view(outputs.shape)).item()
                count += 1

            avg_loss = val_loss / count
            avg_metric = val_metric / count
            print(f"           Validation Loss: {avg_loss:.4f} - Validation Metric: {avg_metric:.4f}")
            self.visualize_sample(images, outputs, labels)

            # ✅ Compute fuel cost for first prediction
            predicted_matrix = sigmoid_outputs[0].detach().cpu().reshape(10, 10)
            fuel = compute_min_fuel(predicted_matrix.numpy(), threshold=0.3)

            print(f"🛢️  Minimum fuel required to reach Base {predicted_matrix.size(0)-1}: {fuel}")

        self.model.train()

    def visualize_sample(self, images, outputs, labels):
        os.makedirs("visuals", exist_ok=True)
        pred = torch.sigmoid(outputs[0]).detach().cpu().reshape(10, 10)
        pred = (pred > 0.5).float()
        truth = labels[0].detach().cpu().reshape(10, 10)

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(pred, cmap="viridis")
        axs[0].set_title("Prediction")
        axs[1].imshow(truth, cmap="viridis")
        axs[1].set_title("Ground Truth")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("visuals/epoch_visual.png")
        plt.close()

    def compute_pos_weight(self, dataset):
        all_labels = torch.cat([label.unsqueeze(0) for _, label in dataset], dim=0)
        positive = all_labels.sum().item()
        negative = all_labels.numel() - positive
        pos_weight_value = negative / positive if positive > 0 else 1.0
        return torch.tensor([pos_weight_value], dtype=torch.float32)
