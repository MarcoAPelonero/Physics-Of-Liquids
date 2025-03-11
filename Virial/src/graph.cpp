#include "graph.hpp"

NDGraph::NDGraph() {
    // Empty graph.
}

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
