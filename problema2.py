import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq
import time

class UnionFind:
    def __init__(self, M, N):
        self.parent = {(i, j): (i, j) for i in range(M) for j in range(N)}

    def find(self, cell):
        if self.parent[cell] != cell:
            self.parent[cell] = self.find(self.parent[cell])
        return self.parent[cell]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            return True
        return False

def generar_laberinto_kruskal(M, N):
    expanded = np.ones((2*M + 1, 2*N + 1))
    for m in range(M):
        for n in range(N):
            expanded[2*m+1][2*n+1] = 0

    walls = []
    for m in range(M):
        for n in range(N):
            if m < M - 1:
                walls.append(((m, n), (m+1, n)))
            if n < N - 1:
                walls.append(((m, n), (m, n+1)))

    random.shuffle(walls)
    uf = UnionFind(M, N)

    for cell1, cell2 in walls:
        if uf.union(cell1, cell2):
            x1, y1 = cell1
            x2, y2 = cell2
            wx = x1 + x2 + 1
            wy = y1 + y2 + 1
            expanded[wx][wy] = 0

    return expanded

def resolver_laberinto(expanded, inicio=(1, 1), fin=(119, 159)):
    def vecinos(pos):
        x, y = pos
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < expanded.shape[0] and 0 <= ny < expanded.shape[1]:
                if expanded[nx][ny] == 0:
                    yield (nx, ny)

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # A* desde inicio hasta fin
    start = time.time()
    heap = [(0 + manhattan(inicio, fin), 0, inicio)]
    came_from = {inicio: None}
    cost_so_far = {inicio: 0}
    explorados = set()

    while heap:
        _, costo, actual = heapq.heappop(heap)
        if actual in explorados:
            continue
        explorados.add(actual)
        if actual == fin:
            break
        for vecino_ in vecinos(actual):
            new_cost = cost_so_far[actual] + 1
            if vecino_ not in cost_so_far or new_cost < cost_so_far[vecino_]:
                cost_so_far[vecino_] = new_cost
                priority = new_cost + manhattan(vecino_, fin)
                heapq.heappush(heap, (priority, new_cost, vecino_))
                came_from[vecino_] = actual

    end = time.time()

    # Reconstrucción del camino
    camino = []
    if fin in came_from:
        actual = fin
        while actual != inicio:
            camino.append(actual)
            actual = came_from[actual]
        camino.append(inicio)
        camino.reverse()

    print("Longitud del camino:", len(camino))
    print("Nodos explorados:", len(explorados))
    print("Tiempo de ejecución:", end - start)

    # Visualización
    fig, ax = plt.subplots()
    ax.imshow(expanded, cmap='gray')
    for (x, y) in explorados:
        ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='khaki', alpha=0.5))
    for (x, y) in camino:
        ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='limegreen', alpha=0.8))
    ax.plot(inicio[1], inicio[0], "bo")  # Inicio
    ax.plot(fin[1], fin[0], "ro")        # Fin
    ax.set_title("Resolución del laberinto con A*")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# === Ejecución del problema 2 ===
if __name__ == "__main__":
    laberinto = generar_laberinto_kruskal(60, 80)
    resolver_laberinto(laberinto, inicio=(1,1), fin=(119,159))