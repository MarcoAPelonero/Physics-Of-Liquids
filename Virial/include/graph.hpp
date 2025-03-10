#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <functional>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <cstdlib>

class Graph {
public:
    int n; // number of vertices
    // Adjacency matrix (n x n), symmetric with no self–loops.
    std::vector<std::vector<bool>> adj;
    
    Graph(int n) : n(n), adj(n, std::vector<bool>(n, false)) { }
    
    // Add an undirected edge between vertices i and j.
    void addEdge(int i, int j) {
        adj[i][j] = true;
        adj[j][i] = true;
    }
    
    bool operator==(const Graph &other) const {
        return n == other.n && adj == other.adj;
    }
};

namespace GraphUtils {

    // Depth–first search to check connectivity.
    bool isConnected(const Graph &g) {
        std::vector<bool> visited(g.n, false);
        std::function<void(int)> dfs = [&](int v) {
            visited[v] = true;
            for (int w = 0; w < g.n; ++w) {
                if (g.adj[v][w] && !visited[w])
                    dfs(w);
            }
        };
        dfs(0);
        return std::all_of(visited.begin(), visited.end(), [](bool v){ return v; });
    }
    
    // Check biconnectivity (i.e. no articulation points).
    // For n < 3, a connected graph is taken as biconnected.
    bool isBiconnected(const Graph &g) {
        if (g.n < 3) return isConnected(g);
        std::vector<int> disc(g.n, -1), low(g.n, -1), parent(g.n, -1);
        int time = 0;
        bool hasArticulation = false;
        
        std::function<void(int)> dfsAP = [&](int u) {
            disc[u] = low[u] = time++;
            int children = 0;
            for (int v = 0; v < g.n; ++v) {
                if (g.adj[u][v]) {
                    if (disc[v] == -1) {
                        ++children;
                        parent[v] = u;
                        dfsAP(v);
                        low[u] = std::min(low[u], low[v]);
                        if (parent[u] == -1 && children > 1)
                            hasArticulation = true;
                        if (parent[u] != -1 && low[v] >= disc[u])
                            hasArticulation = true;
                    } else if (v != parent[u]) {
                        low[u] = std::min(low[u], disc[v]);
                    }
                }
            }
        };
        dfsAP(0);
        return !hasArticulation;
    }
    
    // Compute the automorphism count of a graph by brute force.
    // (Only feasible for small n.)
    int automorphismCount(const Graph &g) {
        int count = 0;
        std::vector<int> perm(g.n);
        std::iota(perm.begin(), perm.end(), 0);
        do {
            bool isAuto = true;
            for (int i = 0; i < g.n && isAuto; ++i) {
                for (int j = 0; j < g.n && isAuto; ++j) {
                    if (g.adj[i][j] != g.adj[perm[i]][perm[j]])
                        isAuto = false;
                }
            }
            if (isAuto) ++count;
        } while (std::next_permutation(perm.begin(), perm.end()));
        return count;
    }
    
    // Generate all graphs on n vertices (using brute force over 2^(n*(n–1)/2) possibilities)
    // that are connected and biconnected (i.e. irreducible in the Mayer expansion).
    inline std::vector<Graph> generateIrreducibleGraphs(int n) {
        std::vector<Graph> graphs;
        int numEdges = n*(n-1)/2;
        int total = 1 << numEdges;
        // Use a fixed ordering for the potential edges.
        for (int mask = 0; mask < total; ++mask) {
            Graph g(n);
            int edgeIndex = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = i+1; j < n; ++j) {
                    if (mask & (1 << edgeIndex))
                        g.addEdge(i, j);
                    ++edgeIndex;
                }
            }
            if (!isConnected(g)) continue;
            if (!isBiconnected(g)) continue;
            graphs.push_back(g);
        }
        return graphs;
    }
}

#endif // GRAPH_HPP
