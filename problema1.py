# Problema 1: 
# Generacion aleatoria de laberintos

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# 1.1 Algoritmo de Prim
class Prim:
    def __init__(self,M,N, animate):
        # Grilla
        self.grid = np.ones((M,N)) # Visitados o no visitados
        self.expanded_grid = np.ones((2*M+1,2*N+1)) # Grilla de laberinto final
        self.m = M
        self.n = N
        # Random Start:
        start = (random.randint(0,M-1),random.randint(0,N-1)) # Eleecion al azar del inicio
        self.grid[start[0]][start[1]] = 0 # Se toma como visitado
        self.expanded_grid[2*start[0]+1,2*start[1]+1]= 3 # Se pinta azul
        self.walls = self.get_walls(start[0],start[1],M, N) # Se agregan sus paredes
        # Plot
        self.anim = animate
        if animate>0:
            colors = ['white','black','red','blue']
            self.cmap = ListedColormap(colors)
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Prims's Algorithm - Animated")
            self.img = self.ax.imshow(self.expanded_grid, cmap=self.cmap, vmin=0, vmax=3)
            plt.axis('off')
        
    def get_walls(self, x, y, M, N):
        # Obtiene las parades de todas las direcciones adyacentes a x,y
        walls = []
        if x<M-1:
            if self.grid[x+1][y]>0:
                walls.append(((x,y),(x+1,y)))
        if y<N-1:
            if self.grid[x][y+1]>0:
                walls.append(((x,y),(x,y+1)))      
        if x>0:
            if self.grid[x-1][y]>0:
                walls.append(((x,y),(x-1,y)))
        if y>0:
            if self.grid[x][y-1]>0:
                walls.append(((x,y),(x,y-1)))
        return walls

    def gen_maze(self):
        # Ciclo de generacion
        while (len(self.walls))>0:
            # Se pintan rojo las posibles celdas siguientes
            for w in self.walls:
                if self.grid[w[1][0]][w[1][1]] > 0:
                    self.expanded_grid[2*w[1][0]+1][2*w[1][1]+1] = 2
            
            # Se hace pop a una pared al azar de walls
            rand_wall = self.walls.pop(random.randint(0,len(self.walls)-1))

            # Si no ha sido visitado...
            if self.grid[rand_wall[1][0]][rand_wall[1][1]]>0:
                # Se marca origen y destino como visitados
                self.grid[rand_wall[0][0]][rand_wall[0][1]]=0
                self.grid[rand_wall[1][0]][rand_wall[1][1]]=0
                
                # Se libera la pared en la grilla de laberinto
                self.expanded_grid[2*rand_wall[1][0]+1][2*rand_wall[1][1]+1] = 0

                # Se pinta blanco la celda destino en la grilla del laberinto
                diff = (
                    rand_wall[1][0] - rand_wall[0][0],
                    rand_wall[1][1] - rand_wall[0][1]
                )
                self.expanded_grid[
                    2*rand_wall[0][0] + diff[0] + 1,
                    2*rand_wall[0][1] + diff[1] + 1
                ] = 0
                
                # Se obtienen paredes de la celda destino
                possible_walls = self.get_walls(
                    rand_wall[1][0],
                    rand_wall[1][1],
                    self.m,self.n
                )
                # Se agregan las que no existan
                for w in possible_walls:
                    if w not in self.walls:
                        self.walls.append(w) 
                if self.anim==2:
                    self.img.set_data(self.expanded_grid)
                    plt.draw()
                    plt.pause(0.01)
        
        if self.anim>0:
            self.img = self.ax.imshow(self.expanded_grid, cmap=self.cmap, vmin=0,vmax=3)
            plt.axis('off')
            self.ax.set_title("Prim Algorithm - Finished!")
            plt.ioff()
            plt.show()
        return self.expanded_grid

# 1.2 Algoritmo de Kruskal
class Kruskal:
    def __init__(self, M, N, animate):
        # Init grilla de laberinto
        self.grid = np.ones((2*M+1,2*N+1))
        
        # Plot
        self.anim = animate
        if animate>0:
            colors = ['white','black']
            cmap = ListedColormap(colors)
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Kruskal's Algorithm - Animated")
            self.img = self.ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=1)
            plt.axis('off')
        
        # Se agregan todas las paredes posibles entre 2 celdas
        walls = []
        for m in range(M):
            for n in range(N):
                if m<M-1:
                    walls.append(((m,n),(m+1,n)))
                if n<N-1:
                    walls.append(((m,n),(m,n+1)))
        self.walls =walls
        # Se aleatorizan las paredes (asegura un nuevo laberinto cada ejecucion)
        
        random.shuffle(self.walls)
        # Union Find Parent
        self.uf_parent = {(i, j): (i, j) for i in range(M) for j in range(N)}


    def uf_find(self, cell):
        # Union find todas las celdas de una particion
        if self.uf_parent[cell] != cell:
            self.uf_parent[cell] = self.uf_find(self.uf_parent[cell])
        return self.uf_parent[cell]
    
    def uf_union(self, a, b):
        # Union find: conectar particiones
        ra = self.uf_find(a)
        rb = self.uf_find(b)
        if ra != rb:
            self.uf_parent[rb] = ra
            return True
        return False 
    
    def gen_maze(self):
        # Ciclo principal
        for cell1, cell2 in self.walls:
            # Se liberan celdas origen y destino
            self.grid[2*cell1[0]+1][2*cell1[1]+1] = 0
            self.grid[2*cell2[0]+1][2*cell2[1]+1] = 0
            
            if self.uf_union(cell1, cell2): # Si no pertenecen a la misma particion
                # Encontrar pared y liberarla
                x1, y1 = cell1
                x2, y2 = cell2
                wx = x1 + x2 + 1
                wy = y1 + y2 + 1
                self.grid[wx][wy] = 0  # Remove wall
                if self.anim==2:
                    self.img.set_data(self.grid)
                    plt.draw()
                    plt.pause(0.00001)
    
        if self.anim==1:
            self.img.set_data(self.grid)
            plt.draw()
        plt.ioff()
        plt.show()
        return self.grid

def generate_maze(M,N, method, do_animation):
    if method == "kruskal":
        kr = Kruskal(M,N,do_animation)
        maze = kr.gen_maze()
    elif method =="prim":
        pr = Prim(M,N,do_animation)
        maze = pr.gen_maze()
    else:
        print("Method not defined")
    return maze

def maze_gen_example():
    M = int(input("M: "))
    N = int(input("N: "))
    method = input("Metodo: ")
    do_animation = int(input("Animacion (0: Ninguna, 1: Solo Resultado, 2: Animar proceso):"))
    maze = generate_maze(M,N,method, do_animation)
    print(maze)

maze_gen_example()
# generate_maze(40,40,"kruskal",True)
# generate_maze(40,40,"kruskal",False)
# generate_maze(40,40,"prim",True)
# generate_maze(40,40,"prim",False)