import torch
from torch.utils.data import DataLoader



def adjacency_matrix_accuracy(preds, targets, threshold=0.5):
    # preds expected to be sigmoid outputs already
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.mean()


class Trainer:
    # ... your __init__ unchanged ...

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
                labels = labels.to(self.device).float()  # ensure float

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.loss_fn(outputs, labels.view(outputs.shape))
                loss.backward()
                self.optimizer.step()

                sigmoid_outputs = torch.sigmoid(outputs)

                # Debug prints (remove after confirming shapes/values)
                if count == 0 and epoch == 0:
                    print("sigmoid_outputs min/max:", sigmoid_outputs.min().item(), sigmoid_outputs.max().item())
                    print("labels min/max:", labels.min().item(), labels.max().item())
                    print("sigmoid_outputs shape:", sigmoid_outputs.shape)
                    print("labels shape:", labels.shape)

                epoch_loss += loss.item()
                total_metric += self.metric_fn(sigmoid_outputs, labels.view(outputs.shape)).item()

                count += 1

            avg_loss = epoch_loss / count
            avg_metric = total_metric / count
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f} - Train Metric: {avg_metric:.4f}")

            self.validate(val_loader)

    # ... your validate unchanged ...
