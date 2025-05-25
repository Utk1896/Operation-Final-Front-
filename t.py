from data.graph_dataset import GraphDataset
from torchvision.transforms.functional import to_pil_image
csv_path = "/home/Utkarsh/pCLUB/data/graphs.csv"
image_folder = "/home/Utkarsh/pCLUB/data/images"

dataset = GraphDataset(csv_path, image_folder)
print(f"Dataset size: {len(dataset)}")

image, adjacency = dataset[0]
print("Image type:", type(image), "Adjacency shape:", adjacency.shape)

# Convert tensor to PIL and show
to_pil_image(image).show()
