import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_adjacency_matrix(matrix, title="Adjacency Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='Greys', interpolation='none')
    plt.title(title)
    plt.xlabel('To Node')
    plt.ylabel('From Node')
    plt.colorbar(label='Edge Presence')
    plt.grid(False)
    plt.show()

def visualize_graph_from_adj_matrix(adj_matrix, title="Graph"):
    G = nx.DiGraph()
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
        for j in range(num_nodes):
            if adj_matrix[i][j] > 0.5:  # treat > 0.5 as edge presence
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, arrowstyle='-|>', arrowsize=20)
    plt.title(title)
    plt.show()

def compare_predictions(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).astype(int)
    target = target.astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pred_binary, cmap='Greys')
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("To")
    axes[0].set_ylabel("From")

    axes[1].imshow(target, cmap='Greys')
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("To")
    axes[1].set_ylabel("From")

    plt.tight_layout()
    plt.show()

def visualize_training(history):
    epochs = len(history['train_loss'])
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Train')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Val')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_metric'], label='Train Acc')
    plt.plot(range(1, epochs+1), history['val_metric'], label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
