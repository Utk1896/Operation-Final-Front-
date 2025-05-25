import networkx as nx
from collections import deque

def compute_min_fuel(adjacency, threshold=0.5):
    N = adjacency.shape[0]
    G_forward = nx.DiGraph()
    G_reverse = nx.DiGraph()

    for i in range(N):
        for j in range(N):
            if adjacency[i][j] > threshold:
                G_forward.add_edge(i, j)
                G_reverse.add_edge(j, i)

    dq = deque()
    visited = [[False] * 2 for _ in range(N)] 
    dq.append((0, 0, 0)) 

    while dq:
        node, flipped, cost = dq.popleft()

        if node == N - 1:
            return cost

        if visited[node][flipped]:
            continue
        visited[node][flipped] = True

        G = G_reverse if flipped else G_forward
        for neighbor in G.neighbors(node):
            dq.appendleft((neighbor, flipped, cost + 1))

        if not visited[node][1 - flipped]:
            dq.append((node, 1 - flipped, cost + N))

    return -1  
