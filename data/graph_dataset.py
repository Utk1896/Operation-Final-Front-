import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .preprocess import preprocess_image  # import your updated function


preprocess_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),  # adjust if needed
    transforms.ToTensor(),
])
class GraphDataset(Dataset):
    def __init__(self, csv_file, image_folder, max_nodes=10, transform=None):
        self.graphs = []
        self.image_folder = image_folder
        self.max_nodes = max_nodes
        self.transform = transform

        # Manually parse CSV instead of pd.read_csv
        with open(csv_file, 'r') as f:
            lines = f.readlines()[1:]  # skip header line
            for line in lines:
                parts = line.strip().split(',')
                graph_id = int(parts[0])
                num_nodes = int(parts[1])
                adjacency_flat = list(map(int, parts[2:] ))
                self.graphs.append((graph_id, num_nodes, adjacency_flat))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph_id, num_nodes, adjacency_flat = self.graphs[idx]

        # Pad adjacency matrix to max_nodes x max_nodes
        adjacency = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        actual_adj = torch.tensor(adjacency_flat, dtype=torch.float32).reshape(num_nodes, num_nodes)
        adjacency[:num_nodes, :num_nodes] = actual_adj

        image_path = os.path.join(self.image_folder, f"graph_{graph_id}.png")
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Warning: Image {image_path} not found, returning blank image")
            image = Image.new('RGB', (64, 64))  # keep RGB to match 3 channels

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Flatten adjacency matrix here
        adjacency = adjacency.view(-1)  # flatten to (max_nodes*max_nodes,)

        return image, adjacency


# Now instantiate the dataset with the preprocessing transform
dataset = GraphDataset(
    csv_file="/home/Utkarsh/pCLUB/data/graphs.csv",
    image_folder="/home/Utkarsh/pCLUB/data/images",
    max_nodes=10,
    transform=preprocess_image  # your preprocessing function
)
