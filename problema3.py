import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time
import random
from collections import deque
import heapq

# === Generador de laberintos ===
def generar_laberinto(filas, columnas, densidad=0.3):
    return np.random.choice([0, 1], size=(filas, columnas), p=[1 - densidad, densidad])

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def vecinos(pos, lab):
    x, y = pos
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < lab.shape[0] and 0 <= ny < lab.shape[1] and lab[nx][ny] == 0:
            yield (nx, ny)

# === Algoritmos de búsqueda ===
def bfs(lab, inicio, fin):
    start = time.time()
    queue = deque([inicio])
    came_from = {inicio: None}
    explorados = set()
    
    while queue:
        actual = queue.popleft()
        explorados.add(actual)
        if actual == fin:
            break
        for vecino_ in vecinos(actual, lab):
            if vecino_ not in came_from:
                came_from[vecino_] = actual
                queue.append(vecino_)
    
    end = time.time()
    camino = reconstruir_camino(came_from, inicio, fin)
    return camino, explorados, end - start

def dfs(lab, inicio, fin):
    start = time.time()
    stack = [inicio]
    came_from = {inicio: None}
    explorados = set()
    
    while stack:
        actual = stack.pop()
        if actual in explorados:
            continue
        explorados.add(actual)
        if actual == fin:
            break
        for vecino_ in vecinos(actual, lab):
            if vecino_ not in came_from:
                came_from[vecino_] = actual
                stack.append(vecino_)
    
    end = time.time()
    camino = reconstruir_camino(came_from, inicio, fin)
    return camino, explorados, end - start

def dijkstra(lab, inicio, fin):
    start = time.time()
    heap = [(0, inicio)]
    came_from = {inicio: None}
    cost_so_far = {inicio: 0}
    explorados = set()
    
    while heap:
        costo, actual = heapq.heappop(heap)
        explorados.add(actual)
        if actual == fin:
            break
        for vecino_ in vecinos(actual, lab):
            new_cost = cost_so_far[actual] + 1
            if vecino_ not in cost_so_far or new_cost < cost_so_far[vecino_]:
                cost_so_far[vecino_] = new_cost
                heapq.heappush(heap, (new_cost, vecino_))
                came_from[vecino_] = actual
    
    end = time.time()
    camino = reconstruir_camino(came_from, inicio, fin)
    return camino, explorados, end - start

def astar(lab, inicio, fin):
    start = time.time()
    heap = [(0 + manhattan(inicio, fin), 0, inicio)]
    came_from = {inicio: None}
    cost_so_far = {inicio: 0}
    explorados = set()
    
    while heap:
        _, costo, actual = heapq.heappop(heap)
        explorados.add(actual)
        if actual == fin:
            break
        for vecino_ in vecinos(actual, lab):
            new_cost = cost_so_far[actual] + 1
            if vecino_ not in cost_so_far or new_cost < cost_so_far[vecino_]:
                cost_so_far[vecino_] = new_cost
                priority = new_cost + manhattan(vecino_, fin)
                heapq.heappush(heap, (priority, new_cost, vecino_))
                came_from[vecino_] = actual
    
    end = time.time()
    camino = reconstruir_camino(came_from, inicio, fin)
    return camino, explorados, end - start

def reconstruir_camino(came_from, inicio, fin):
    if fin not in came_from:
        return []
    path = []
    actual = fin
    while actual != inicio:
        path.append(actual)
        actual = came_from[actual]
    path.append(inicio)
    path.reverse()
    return path

# === Ranking ===
def rank_algoritmos(row):
    algoritmos = ['BFS', 'DFS', 'Dijkstra', 'A*']
    comparables = []
    for alg in algoritmos:
        l = row[f'{alg}_Longitud']
        if l > 0:
            t = row[f'{alg}_Tiempo']
            n = row[f'{alg}_Nodos']
            comparables.append((alg, l, t, n))
    comparables.sort(key=lambda x: (x[1], x[2], x[3]))
    ranking = {alg: 4 for alg in algoritmos}
    for i, (alg, *_rest) in enumerate(comparables):
        ranking[alg] = i + 1
    return pd.Series(ranking, index=[f'Rank_{alg}' for alg in algoritmos])

# === Visualización ===
def mostrar_simulacion(lab, inicio, fin, soluciones, sim_num, ranking_row):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    algoritmos = ['BFS', 'DFS', 'Dijkstra', 'A*']
    for i, alg in enumerate(algoritmos):
        ax = axes[i]
        ax.set_title(f"{alg} (Rank {ranking_row[f'Rank_{alg}']})")
        ax.imshow(lab, cmap='gray', interpolation='none')
        for (x, y) in soluciones[alg]['explorados']:
            ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='khaki', alpha=0.5))
        for (x, y) in soluciones[alg]['camino']:
            ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='limegreen', alpha=0.8))
        ax.plot(inicio[1], inicio[0], "bo")
        ax.plot(fin[1], fin[0], "ro")
        ax.axis('off')
    fig.suptitle(f"Simulación #{sim_num}", fontsize=16)
    plt.tight_layout()
    plt.show()

# === Simulaciones ===
resultados = []
for sim in range(1, 26):
    while True:
        lab = generar_laberinto(45, 55)
        libres = np.argwhere(lab == 0)
        if len(libres) < 2: continue
        inicio = tuple(random.choice(libres))
        posibles = [p for p in libres if manhattan(p, inicio) >= 10]
        if posibles:
            fin = tuple(random.choice(posibles))
            break

    camino_bfs, explorados_bfs, t_bfs = bfs(lab, inicio, fin)
    camino_dfs, explorados_dfs, t_dfs = dfs(lab, inicio, fin)
    camino_dijkstra, explorados_dijkstra, t_dijkstra = dijkstra(lab, inicio, fin)
    camino_astar, explorados_astar, t_astar = astar(lab, inicio, fin)

    row = {
        'Sim': sim,
        'BFS_Nodos': len(explorados_bfs), 'BFS_Tiempo': t_bfs, 'BFS_Longitud': len(camino_bfs),
        'DFS_Nodos': len(explorados_dfs), 'DFS_Tiempo': t_dfs, 'DFS_Longitud': len(camino_dfs),
        'Dijkstra_Nodos': len(explorados_dijkstra), 'Dijkstra_Tiempo': t_dijkstra, 'Dijkstra_Longitud': len(camino_dijkstra),
        'A*_Nodos': len(explorados_astar), 'A*_Tiempo': t_astar, 'A*_Longitud': len(camino_astar)
    }

    soluciones = {
        'BFS': {'camino': camino_bfs, 'explorados': explorados_bfs},
        'DFS': {'camino': camino_dfs, 'explorados': explorados_dfs},
        'Dijkstra': {'camino': camino_dijkstra, 'explorados': explorados_dijkstra},
        'A*': {'camino': camino_astar, 'explorados': explorados_astar}
    }

    ranking = rank_algoritmos(row)
    row.update(ranking.to_dict())
    resultados.append(row)
    mostrar_simulacion(lab, inicio, fin, soluciones, sim, ranking)
    df_temp = pd.DataFrame([row])
    print(df_temp[['Sim'] + [col for col in df_temp.columns if 'Rank' in col]])

# === Resultados finales ===
df = pd.DataFrame(resultados)
print("\n🏁 Ranking Promedio Final:\n")
ranking_promedio = df[[f'Rank_{alg}' for alg in ['BFS', 'DFS', 'Dijkstra', 'A*']]].mean().sort_values()
print(ranking_promedio)
