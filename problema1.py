# Generacion aleatoria de laberintos
import numpy as np
import random
import matplotlib.pyplot as plt

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
    
    grid = np.zeros((M,N))
    kruskal(M,N)
    
    
def kruskal(M, N):
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

    for cell1, cell2 in walls:
        if uf.union(cell1, cell2):
            # Determine the wall between cell1 and cell2 and remove it
            x1, y1 = cell1
            x2, y2 = cell2
            wx = x1 + x2 + 1
            wy = y1 + y2 + 1
            expanded[wx][wy] = 0  # Remove wall
    
    plt.imshow(expanded, cmap='Greys', interpolation='nearest')
    plt.axis('off')  # Turn off axes
    plt.show()
    
generate(60,80)