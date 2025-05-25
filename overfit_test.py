import torch
from models.extractor import GraphExtractor
from data.graph_dataset import GraphDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = GraphDataset(
    "/home/Utkarsh/pCLUB/data/graphs.csv",
    "/home/Utkarsh/pCLUB/data/images",
    max_nodes=10,
    transform=transform
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)
image, label = next(iter(loader))
model = GraphExtractor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image, label = image.to(device), label.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label.view(output.shape))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
