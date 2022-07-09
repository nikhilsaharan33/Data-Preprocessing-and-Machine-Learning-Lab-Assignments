# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:24:48 2021

@author: nikhil
"""
import queue
import sys
# For getting input from input.txt file
sys.stdin = open('input.txt', 'r')

class Graph:
    def __init__(self, nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(nVertices)] for j in range(nVertices)]

    def addEdge(self, v1, v2):
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1

    def __dfsHelper(self, sv, visited):
        print(sv, end=" ")
        visited[sv] = True
        for i in range(self.nVertices):
            if (self.adjMatrix[sv][i] > 0 and not visited[i]):
                self.__dfsHelper(i, visited)

    def dfs(self):
        visited = [False for i in range(self.nVertices)]
        for v in range(self.nVertices):
            if (not visited[v]):
                self.__dfsHelper(v, visited)
        print()

    def __bfsHelper(self, sv, visited):
        if (self.nVertices == 0):
            return
        q = queue.Queue()
        q.put(sv)
        visited[sv] = True
        while (not q.empty()):
            v = q.get()
            print(v, end=" ")
            for i in range(self.nVertices):
                if (self.adjMatrix[v][i] > 0 and not visited[i]):
                    q.put(i)
                    visited[i] = True

    def bfs(self):
        visited = [False] * self.nVertices
        for v in range(self.nVertices):
            if (not visited[v]):
                self.__bfsHelper(v, visited)
        print()

    def __hasPathHelper(self, v1, v2, visited):
        visited[v1] = True
        for i in range(self.nVertices):
            if (self.adjMatrix[v1][i] > 0 and not visited[i]):
                if (i == v2):
                    return True
                if (self.__hasPathHelper(i, v2, visited)):
                    return True
        return False

    def hasPath(self, v1, v2):
        visited = [False for i in range(self.nVertices)]
        return self.__hasPathHelper(v1, v2, visited)

    def __getPathHelperdfs(self, v1, v2, visited):
        visited[v1] = True
        for i in range(self.nVertices):
            if (self.adjMatrix[v1][i] > 0 and not visited[i]):
                if (i == v2):
                    return [v2, v1]

                path = self.__getPathHelperdfs(i, v2, visited)
                if (path is not None):
                    path.append(v1)
                    return path
        return None

    def getPathdfs(self, v1, v2):
        if (v1 == v2):
            return [v1, v1]
        visited = [False for i in range(self.nVertices)]
        return self.__getPathHelperdfs(v1, v2, visited)

    def __getPathHelperbfs(self, parent, v2):
        path = [v2]
        v = v2
        while (parent[v] != -1):
            path.append(parent[v])
            v = parent[v]
        return path

    def getPathbfs(self, v1, v2):
        if (v1 == v2):
            return [v2, v1]
        visited = [False] * self.nVertices
        parent = [-1] * self.nVertices
        q = queue.Queue()
        q.put(v1)
        visited[v1] = True
        while (not q.empty()):
            v = q.get()
            for i in range(self.nVertices):
                if (self.adjMatrix[v][i] > 0 and not visited[i]):
                    parent[i] = v
                    q.put(i)
                    visited[i] = True
                    if (i == v2):
                        return self.__getPathHelperbfs(parent, v2)

        return None

    def __isConnectedHelper(self, sv, visited):
        visited[sv] = True
        for i in range(self.nVertices):
            if (self.adjMatrix[sv][i] > 0 and not visited[i]):
                self.__isConnectedHelper(i, visited)

    def isConnected(self):
        visited = [False for i in range(self.nVertices)]
        self.__isConnectedHelper(0, visited)
        for i in visited:
            if (not i):
                return False
        return True

    def removeEdge(self, v1, v2):
        if (self.adjMatrix[v1][v2] == 0):
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def containsEdge(self, v1, v2):
        return True if self.adjMatrix[v1][v2] > 0 else False

    def __str__(self):
        return str(self.adjMatrix)


V, E = (int(x) for x in input().split())
G = Graph(V)
'''
6 6
0 3
0 2
0 1
1 5
2 5
2 4
0 4
'''
for i in range(E):
    v1, v2 = (int(x) for x in input().split())
    G.addEdge(v1, v2)
v1, v2 = (int(x) for x in input().split())
path = G.getPathbfs(v1, v2)
if path is not None:
    print(*path)