#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;

class Graph {
public:
    int n;  
    vector<vector<bool>> adj;

    Graph(int n) : n(n), adj(n, vector<bool>(n, false)) {}

    void addEdge(int u, int v) {
        adj[u][v] = true;
        adj[v][u] = true;
    }

    // Checks if the graph is connected using BFS.
    bool isConnected() const {
        vector<bool> visited(n, false);
        queue<int> q;
        q.push(0);
        visited[0] = true;
        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            for (int i = 0; i < n; i++) {
                if (adj[cur][i] && !visited[i]) {
                    visited[i] = true;
                    q.push(i);
                }
            }
        }
        for (bool v : visited) {
            if (!v)
                return false;
        }
        return true;
    }

    // Helper function for DFS to check biconnectivity.
    // Returns false if an articulation point is found.
    bool isBiconnectedUtil(int u, vector<bool>& visited, vector<int>& disc,
                           vector<int>& low, int parent, int &time) const {
        visited[u] = true;
        disc[u] = low[u] = ++time;
        int children = 0;
        for (int v = 0; v < n; v++) {
            if (adj[u][v]) {
                if (!visited[v]) {
                    children++;
                    if (!isBiconnectedUtil(v, visited, disc, low, u, time))
                        return false;
                    low[u] = min(low[u], low[v]);
                    // If u is not root and low value of one of its children is
                    // greater or equal to discovery value of u, then u is an articulation point.
                    if (parent != -1 && low[v] >= disc[u])
                        return false;
                } else if (v != parent) {  // a back edge exists
                    low[u] = min(low[u], disc[v]);
                }
            }
        }
        // Special case: if u is root and has more than one child, it's an articulation point.
        if (parent == -1 && children > 1)
            return false;
        return true;
    }

    // Checks if the graph is "irreducible" (here defined as biconnected).
    bool isIrreducible() const {
        if (!isConnected())
            return false;
        vector<bool> visited(n, false);
        vector<int> disc(n, 0), low(n, 0);
        int time = 0;
        return isBiconnectedUtil(0, visited, disc, low, -1, time);
    }

    // Compute a canonical label for the graph by brute-forcing over all vertex permutations.
    // The label is obtained by considering the upper-triangle of the permuted adjacency matrix.
    std::string canonicalLabel() const {
        vector<int> perm(n);
        for (int i = 0; i < n; i++)
            perm[i] = i;
        string best;
        bool first = true;
        do {
            string s;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    s.push_back(adj[perm[i]][perm[j]] ? '1' : '0');
                }
            }
            if (first || s < best) {
                best = s;
                first = false;
            }
        } while (next_permutation(perm.begin(), perm.end()));
        return best;
    }

    // Returns the list of edges as an array of pairs [u, v] (with u < v).
    vector<vector<int>> getEdges() const {
        vector<vector<int>> edges;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (adj[i][j])
                    edges.push_back({i, j});
            }
        }
        return edges;
    }
};

// Generates graphs on n vertices, filters them according to 'option',
// groups isomorphic graphs (by their canonical label), and writes a JSON file.
// 'option' values:
//   0 - all graphs (up to isomorphism),
//   1 - only connected graphs,
//   2 - only irreducible (biconnected) graphs.
void generateIndependentGraphs(int n, int option, const string &filename) {
    int m = n * (n - 1) / 2; // number of possible edges in an undirected graph
    // Map: canonical label -> (representative graph, multiplicity count)
    map<string, pair<Graph, int>> canonMap;
    for (int mask = 0; mask < (1 << m); mask++) {
        Graph g(n);
        int edgeIndex = 0;
        // Build graph from bitmask.
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (mask & (1 << edgeIndex))
                    g.addEdge(i, j);
                edgeIndex++;
            }
        }
        // Apply the selected filter.
        if (option == 1 && !g.isConnected())
            continue;
        if (option == 2 && !g.isIrreducible())
            continue;

        // Compute canonical label.
        string canon = g.canonicalLabel();
        auto it = canonMap.find(canon);
        if (it == canonMap.end())
            canonMap.emplace(canon, make_pair(g, 1));
        else
            it->second.second++;
    }

    // Write the independent graphs (one per isomorphism class) to a JSON file.
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Error opening file for writing." << endl;
        return;
    }
    outFile << "[\n";
    bool firstEntry = true;
    for (auto &entry : canonMap) {
        if (!firstEntry)
            outFile << ",\n";
        const string &canon = entry.first;
        Graph &rep = entry.second.first;
        int multiplicity = entry.second.second;
        outFile << "  { \"canonical\": \"" << canon << "\", ";
        outFile << "\"n\": " << rep.n << ", ";
        // Write the edge list.
        vector<vector<int>> edges = rep.getEdges();
        outFile << "\"edges\": [";
        bool firstEdge = true;
        for (auto &e : edges) {
            if (!firstEdge)
                outFile << ", ";
            outFile << "[" << e[0] << "," << e[1] << "]";
            firstEdge = false;
        }
        outFile << "], ";
        outFile << "\"multiplicity\": " << multiplicity << " }";
        firstEntry = false;
    }
    outFile << "\n]\n";
    outFile.close();
    cout << "Independent graphs written to " << filename << endl;
}

int main() {
    int n, option;
    cout << "Enter number of vertices: ";
    cin >> n;
    cout << "Select option:" << endl;
    cout << "0 - All graphs (up to isomorphism)" << endl;
    cout << "1 - Only connected graphs" << endl;
    cout << "2 - Only irreducible (biconnected) graphs" << endl;
    cin >> option;

    generateIndependentGraphs(n, option, "independent_graphs.json");
    return 0;
}
