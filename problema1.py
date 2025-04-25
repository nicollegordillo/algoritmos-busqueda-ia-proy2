# Generacion aleatoria de laberintos
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    
def generate(M,N):
    # prim(M,N, True)
    kruskal(M,N,True)

def get_walls(grid, x, y, M, N):
    walls = []
    if x<M-1:
        if grid[x+1][y]>0:
            walls.append(((x,y),(x+1,y)))
    if y<N-1:
        if grid[x][y+1]>0:
            walls.append(((x,y),(x,y+1)))      
    if x>0:
        if grid[x-1][y]>0:
            walls.append(((x,y),(x-1,y)))
    if y>0:
        if grid[x][y-1]>0:
            walls.append(((x,y),(x,y-1)))
    return walls

def color_start(expanded, x, y):
    for i in range(-1,2):
        for j in range(-1,2):   
            expanded[x+i][y+j] = 3
def prim(M,N,animate):
    grid = np.ones((M,N))
    expanded = np.ones((2*M+1,2*N+1))
    colors = ['white','black','red','blue']
    cmap = ListedColormap(colors)
    start = (random.randint(0,M-1),random.randint(0,N-1))
    # start = (0,0)
    grid[start[0]][start[1]] = 0
    walls = get_walls(grid, start[0],start[1],M, N)
    
    
    if animate:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_title("Prim Algorithm - Animated")
        img = ax.imshow(expanded, cmap=cmap, vmin=0, vmax=3)
        plt.axis('off')
    
    # for i in range(4):
    while len(walls)>0:
        for w in walls:
            if grid[w[1][0]][w[1][1]] > 0:
                expanded[2*w[1][0]+1][2*w[1][1]+1] = 2
        
        rand_wall = walls.pop(random.randint(0,len(walls)-1))
        
        if grid[rand_wall[1][0]][rand_wall[1][1]]>0:

            # Mark cell as visited
            grid[rand_wall[0][0]][rand_wall[0][1]]=0
            grid[rand_wall[1][0]][rand_wall[1][1]]=0
            expanded[2*rand_wall[1][0]+1][2*rand_wall[1][1]+1] = 0

            diff = (
                rand_wall[1][0] - rand_wall[0][0],
                rand_wall[1][1] - rand_wall[0][1]
            )
            expanded[
                2*rand_wall[0][0] + diff[0] + 1,
                2*rand_wall[0][1] + diff[1] + 1
            ] = 0
            
            possible_walls = get_walls(
                grid,rand_wall[1][0],
                rand_wall[1][1],
                M,N
            )
            for w in possible_walls:
                if w not in walls:
                    walls.append(w) 
            color_start(expanded, 2*start[0]+1, 2*start[1]+1)
            if animate:
                img.set_data(expanded)
                plt.draw()
                plt.pause(0.01)
    print("Finished!!")

    if not animate:
        fig, ax = plt.subplots()
        img = ax.imshow(expanded, cmap=cmap, vmin=0,vmax=3)
        plt.axis('off')
    ax.set_title("Prim Algorithm - Finished!")
    plt.ioff()
    plt.show()
        
        
    
def kruskal(M, N, animate):
    expanded = np.ones((2*M + 1,2*N+1))
    for m in range(M):
        for n in range(N):
            expanded[2*m+1][2*n+1] = 0
    

    walls = []
    for m in range(M):
        for n in range(N):
            if m<M-1:
                walls.append(((m,n),(m+1,n)))
            if n<N-1:
                walls.append(((m,n),(m,n+1)))
            
    random.shuffle(walls)
    uf = UnionFind(M, N)


    if animate:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        img = ax.imshow(expanded, cmap='Greys', interpolation='nearest')
        plt.axis('off')
    for cell1, cell2 in walls:
        if uf.union(cell1, cell2):
            # Determine the wall between cell1 and cell2 and remove it
            x1, y1 = cell1
            x2, y2 = cell2
            wx = x1 + x2 + 1
            wy = y1 + y2 + 1
            expanded[wx][wy] = 0  # Remove wall
            if animate:
                img.set_data(expanded)
                plt.draw()
                plt.pause(0.00001)
    
    if not animate:
        fig, ax = plt.subplots()
        img = ax.imshow(expanded, cmap='Greys', interpolation='nearest')
        plt.axis('off')
    plt.ioff()
    plt.show()
    
generate(40,40)