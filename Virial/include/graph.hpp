#ifndef NDGRAPH_HPP
#define NDGRAPH_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <stack>
#include <functional>

struct Node {
    int id;
    Node(int id) : id(id) {}

    bool operator==(const Node &other) const {
        return id == other.id;
    }
};

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

    int degeneracy = 0;

public:
    NDGraph();
    NDGraph(int numNodes, bool complete = true);

    void setDegeneracy(int d) {degeneracy = d;}
    int getDegeneracy() const {return degeneracy;}
    int getNumNodes() const {return nodes.size();}
    int getNumEdges() const {return edges.size();}
    const std::vector<Node>& getNodes() const {return nodes;}
    const std::vector<Edge>& getEdges() const {return edges;}

    int addNode();
    int removeNode(int id);

    void addEdge(int from, int to);
    void removeEdge(int from, int to);

    bool isConnected() const;
    bool isBiconnected() const;

    int getAutomorphismCount() const;

    bool operator==(const NDGraph &other) const;

    void printGraph() const;
};

#endif // NDGRAPH_HPP
