#ifndef NDGRAPH_HPP
#define NDGRAPH_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// A simple Node structure that only stores an id.
struct Node {
    int id;
    Node(int id) : id(id) {}
};

// An Edge represents a connection (e.g., a Mayer function) between two nodes.
struct Edge {
    int from;
    int to;
    Edge(int from, int to) : from(from), to(to) {}
    bool operator==(const Edge &other) const {
        return from == other.from && to == other.to;
    }
};

class NDGraph {
private:
    std::vector<Node> nodes;   // List of nodes
    std::vector<Edge> edges;   // List of edges

public:
    NDGraph();
    NDGraph(int numNodes, bool complete = true);

    int addNode();
    int removeNode(int id);

    void addEdge(int from, int to);
    void removeEdge(int from, int to);

    void printGraph() const;
};

#endif // NDGRAPH_HPP
