#include "graph.hpp"

NDGraph::NDGraph() {}

NDGraph::NDGraph(int numNodes, bool complete) {
    if (numNodes <= 0) {
        throw std::invalid_argument("Number of nodes must be positive.");
    }
    for (int i = 0; i < numNodes; ++i) {
        nodes.emplace_back(i);
    }
    if (complete) {
        for (int i = 0; i < numNodes; ++i) {
            for (int j = i + 1; j < numNodes; ++j) {
                addEdge(i, j);
            }
        }
    }
}

int NDGraph::addNode() {
    int id = nodes.size();
    nodes.emplace_back(id);
    return id;
}

int NDGraph::removeNode(int id) {
    if (id < 0 || id >= static_cast<int>(nodes.size())) {
        throw std::out_of_range("Node index out of range.");
    }
    nodes.erase(nodes.begin() + id);
    return id;
}

void NDGraph::addEdge(int from, int to) {
    if (from < 0 || from >= static_cast<int>(nodes.size()) ||
        to < 0 || to >= static_cast<int>(nodes.size())) {
        throw std::out_of_range("Node index out of range.");
    }
    edges.emplace_back(from, to);
}


void NDGraph::removeEdge(int from, int to) {
    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [from, to](const Edge &e) { return e.from == from && e.to == to; }),
                edges.end());
}

bool NDGraph::isConnected() const {
    int n = nodes.size();
    if(n == 0) return true;

    std::vector<bool> visited(n, false);
    std::stack<int> stack;
    stack.push(nodes[0].id);
    visited[nodes[0].id] = true;

    while (!stack.empty()) {
        int cur = stack.top();
        stack.pop();
        for (const auto &edge : edges) {
            if (edge.from == cur && !visited[edge.to]) {
                visited[edge.to] = true;
                stack.push(edge.to);
            }
            if (edge.to == cur && !visited[edge.from]) {
                visited[edge.from] = true;
                stack.push(edge.from);
            }
        }
    }
    return std::all_of(visited.begin(), visited.end(), [](bool v) { return v; });
}

bool NDGraph::isBiconnected() const {
    int n = nodes.size();
    if (n <= 1)
        return true; // A graph with 0 or 1 nodes is trivially irreducible.

    // 1. Build an adjacency list from the edge list.
    std::vector<std::vector<int>> adj(n);
    for (const auto &edge : edges) {
        // Assuming an undirected graph:
        adj[edge.from].push_back(edge.to);
        adj[edge.to].push_back(edge.from);
    }
    
    // 2. Initialize arrays for DFS.
    std::vector<bool> visited(n, false);  // To track if a vertex has been visited.
    std::vector<int> disc(n, 0);            // Discovery times of vertices.
    std::vector<int> low(n, 0);             // Low values for vertices.
    std::vector<int> parent(n, -1);         // Parent of each vertex in the DFS tree.
    std::vector<bool> isArticulation(n, false); // Flags for articulation points.
    int time = 0;                         // Global time counter for DFS discovery times.
    
    // 3. Define a DFS function using a lambda.
    // This function explores the graph from vertex 'u'.
    std::function<void(int)> dfs = [&](int u) {
        visited[u] = true;
        disc[u] = low[u] = ++time;  // Set discovery time and initialize low value.
        int children = 0;           // Count of children in DFS tree for vertex u.
        
        // Explore all adjacent vertices of u.
        for (int v : adj[u]) {
            if (!visited[v]) {
                children++;         // v is a child of u.
                parent[v] = u;      // Set parent of v.
                dfs(v);             // Recursively perform DFS on v.
                
                // After visiting v, update low value of u.
                low[u] = std::min(low[u], low[v]);
                
                // Check articulation conditions:
                // (a) u is root and has more than one child.
                if (parent[u] == -1 && children > 1)
                    isArticulation[u] = true;
                // (b) u is not root and low value of v is not less than disc[u].
                if (parent[u] != -1 && low[v] >= disc[u])
                    isArticulation[u] = true;
            } 
            // If v is already visited and is not the parent of u, update low[u].
            else if (v != parent[u]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    };
    
    // 4. Start DFS from vertex 0.
    dfs(0);
    
    // 5. Check connectivity: if any vertex hasn't been visited, the graph is not connected.
    for (bool v : visited) {
        if (!v)
            return false;  // If the graph isn't fully connected, it fails our irreducibility test.
    }
    
    // 6. Check if any vertex was found to be an articulation point.
    // If none exist, the graph is irreducible.
    return std::none_of(isArticulation.begin(), isArticulation.end(), [](bool flag) { return flag; });
}



void NDGraph::printGraph() const {
    std::cout << "Graph Nodes: ";
    for (const auto &node : nodes) {
        std::cout << node.id << " ";
    }
    std::cout << "\nGraph Edges:\n";
    for (const auto &edge : edges) {
        std::cout << edge.from << " -- " << edge.to << "\n";
    }
}
