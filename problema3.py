import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time
import random
from collections import deque
import heapq
from matplotlib.animation import FuncAnimation

def animar_exploracion(lab, inicio, fin, explorados, camino, algoritmo):
    fig, ax = plt.subplots()
    ax.imshow(lab, cmap='gray')
    ax.plot(inicio[1], inicio[0], "bo")  # inicio
    ax.plot(fin[1], fin[0], "ro")        # fin
    ax.set_title(f"Exploración paso a paso: {algoritmo}")
    ax.axis('off')

    explorados_list = list(explorados)
    patches = []

    def init():
        return patches

    def update(frame):
        if frame < len(explorados_list):
            x, y = explorados_list[frame]
            rect = mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='khaki', alpha=0.5)
            ax.add_patch(rect)
            patches.append(rect)
        elif frame == len(explorados_list):
            for (x, y) in camino:
                rect = mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='limegreen', alpha=0.8)
                ax.add_patch(rect)
                patches.append(rect)
        return patches

    ani = FuncAnimation(fig, update, frames=len(explorados_list) + 1,
                        init_func=init, blit=True, interval=10, repeat=False)
    plt.show()


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
        if actual in explorados:
            continue
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
        if actual in explorados:
            continue
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

# === Ranking por simulación ===
def rank_algoritmos(row):
    algoritmos = ['BFS', 'DFS', 'Dijkstra', 'A*']
    comparables = []
    for alg in algoritmos:
        l = row[f'{alg}_Longitud']
        if l > 0:
            t = row[f'{alg}_Tiempo']
            n = row[f'{alg}_Nodos']
            comparables.append((alg, l, t, n))
    comparables.sort(key=lambda x: (x[1], x[2], x[3]))  # Primero por longitud, luego tiempo, luego nodos
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
        ax.imshow(lab, cmap='gray')
        for (x, y) in soluciones[alg]['explorados']:
            ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='khaki', alpha=0.5))
        for (x, y) in soluciones[alg]['camino']:
            ax.add_patch(mpatches.Rectangle((y-0.5, x-0.5), 1, 1, color='limegreen', alpha=0.8))
        ax.plot(inicio[1], inicio[0], "bo")  # inicio
        ax.plot(fin[1], fin[0], "ro")        # fin
        ax.axis('off')
    fig.suptitle(f"Simulación #{sim_num}", fontsize=16)
    plt.tight_layout()
    plt.show()

import random
import pandas as pd
import numpy as np

# === Simulaciones ===
# === Simulaciones ===
resultados = []
laberintos = []
soluciones_sim = []
inicios = []
fines = []

sim_a_animar = random.randint(1, 25)

# Funciones necesarias para los algoritmos de búsqueda (asegúrate de tenerlas definidas)
# Estas funciones deberían devolver el camino, los nodos explorados y el tiempo de ejecución

for sim in range(1, 26):
    while True:
        lab = generar_laberinto(45, 55)
        libres = np.argwhere(lab == 0)
        if len(libres) < 2:
            continue
        inicio = tuple(random.choice(libres))
        posibles = [p for p in libres if manhattan(p, inicio) >= 10]
        if posibles:
            fin = tuple(random.choice(posibles))
            break

    # Ejecutar algoritmos
    camino_bfs, explorados_bfs, t_bfs = bfs(lab, inicio, fin)
    camino_dfs, explorados_dfs, t_dfs = dfs(lab, inicio, fin)
    camino_dijkstra, explorados_dijkstra, t_dijkstra = dijkstra(lab, inicio, fin)
    camino_astar, explorados_astar, t_astar = astar(lab, inicio, fin)

    # Guardar resultados
    row = {
        'Sim': sim,
        'BFS_Nodos': len(explorados_bfs), 'BFS_Tiempo': t_bfs, 'BFS_Longitud': len(camino_bfs),
        'DFS_Nodos': len(explorados_dfs), 'DFS_Tiempo': t_dfs, 'DFS_Longitud': len(camino_dfs),
        'Dijkstra_Nodos': len(explorados_dijkstra), 'Dijkstra_Tiempo': t_dijkstra, 'Dijkstra_Longitud': len(camino_dijkstra),
        'A*_Nodos': len(explorados_astar), 'A*_Tiempo': t_astar, 'A*_Longitud': len(camino_astar)
    }

    resultados.append(row)

    laberintos.append(lab.copy())
    soluciones_sim.append({
        'BFS': {'camino': camino_bfs, 'explorados': explorados_bfs},
        'DFS': {'camino': camino_dfs, 'explorados': explorados_dfs},
        'Dijkstra': {'camino': camino_dijkstra, 'explorados': explorados_dijkstra},
        'A*': {'camino': camino_astar, 'explorados': explorados_astar}
    })
    inicios.append(inicio)
    fines.append(fin)

    if sim == sim_a_animar:
        animar_exploracion(lab, inicio, fin, explorados_astar, camino_astar, "A*")

# Crear el DataFrame de resultados
df = pd.DataFrame(resultados)

# Calcular el ranking de todos los algoritmos dentro de cada simulación
for i, row in enumerate(resultados):
    tiempos = {
        'BFS': row['BFS_Tiempo'],
        'DFS': row['DFS_Tiempo'],
        'Dijkstra': row['Dijkstra_Tiempo'],
        'A*': row['A*_Tiempo']
    }
    # Ordenar los tiempos de menor a mayor
    tiempos_ordenados = sorted(tiempos.items(), key=lambda x: x[1])

    # Asignar ranking (1 para el más rápido)
    for rank, (alg, _) in enumerate(tiempos_ordenados, start=1):
        resultados[i][f'Rank_{alg}'] = rank

# Actualizar DataFrame con los rankings corregidos
df = pd.DataFrame(resultados)

# Imprimir los resultados por simulación
for i in range(25):
    sim = i + 1
    lab = laberintos[i]
    soluciones = soluciones_sim[i]
    inicio = inicios[i]
    fin = fines[i]
    row = resultados[i]
    
    # Acceder correctamente a los rankings desde el DataFrame 'df'
    row_df = df.iloc[i]  # Obtener la fila correspondiente de 'df'

    print(f"\n=== Resultados por simulación #{sim} ===")
    print("Algoritmo     Nodos   Tiempo     Longitud   Ranking")
    print(f"BFS        {row['BFS_Nodos']:>8}  {row['BFS_Tiempo']:.6f}  {row['BFS_Longitud']:>9}  {int(row_df['Rank_BFS'])}")
    print(f"DFS        {row['DFS_Nodos']:>8}  {row['DFS_Tiempo']:.6f}  {row['DFS_Longitud']:>9}  {int(row_df['Rank_DFS'])}")
    print(f"Dijkstra   {row['Dijkstra_Nodos']:>8}  {row['Dijkstra_Tiempo']:.6f}  {row['Dijkstra_Longitud']:>9}  {int(row_df['Rank_Dijkstra'])}")
    print(f"A*         {row['A*_Nodos']:>8}  {row['A*_Tiempo']:.6f}  {row['A*_Longitud']:>9}  {int(row_df['Rank_A*'])} ")

    # Mostrar la simulación visualmente
    ranking = pd.Series({k: row_df[k] for k in row_df.index if 'Rank' in k})

    # Mostrar la simulación visualmente
    mostrar_simulacion(lab, inicio, fin, soluciones, sim, ranking)


# === Análisis adicional ===
# Promedio de nodos, tiempo y longitud de los caminos
promedios = {
    'BFS': {
        'Nodos': np.mean([row['BFS_Nodos'] for row in resultados]),
        'Tiempo': np.mean([row['BFS_Tiempo'] for row in resultados]),
        'Longitud': np.mean([row['BFS_Longitud'] for row in resultados])
    },
    'DFS': {
        'Nodos': np.mean([row['DFS_Nodos'] for row in resultados]),
        'Tiempo': np.mean([row['DFS_Tiempo'] for row in resultados]),
        'Longitud': np.mean([row['DFS_Longitud'] for row in resultados])
    },
    'Dijkstra': {
        'Nodos': np.mean([row['Dijkstra_Nodos'] for row in resultados]),
        'Tiempo': np.mean([row['Dijkstra_Tiempo'] for row in resultados]),
        'Longitud': np.mean([row['Dijkstra_Longitud'] for row in resultados])
    },
    'A*': {
        'Nodos': np.mean([row['A*_Nodos'] for row in resultados]),
        'Tiempo': np.mean([row['A*_Tiempo'] for row in resultados]),
        'Longitud': np.mean([row['A*_Longitud'] for row in resultados])
    }
}

print("\n=== Promedios de desempeño de cada algoritmo ===")
for alg, values in promedios.items():
    print(f"{alg}: Nodos = {values['Nodos']:.2f}, Tiempo = {values['Tiempo']:.6f}, Longitud = {values['Longitud']:.2f}")

# Mostrar cuál algoritmo fue el mejor en cada simulación
print("\n=== Mejor algoritmo en cada simulación (por tiempo más bajo) ===")
for i, row in enumerate(resultados):
    sim = i + 1
    min_time_alg = min([('BFS', row['BFS_Tiempo']),
                        ('DFS', row['DFS_Tiempo']),
                        ('Dijkstra', row['Dijkstra_Tiempo']),
                        ('A*', row['A*_Tiempo'])], key=lambda x: x[1])
    print(f"Simulación {sim}: Mejor algoritmo = {min_time_alg[0]} con tiempo {min_time_alg[1]:.6f}")

# Ranking Promedio Final
ranking_promedio = df[[f'Rank_{alg}' for alg in ['BFS', 'DFS', 'Dijkstra', 'A*']]].mean().sort_values()
print("\n=== Ranking Promedio Final en las 25 Simulaciones ===")
print(ranking_promedio)

# Guardar CSV
df.to_csv("resultados_busqueda.csv", index=False)
