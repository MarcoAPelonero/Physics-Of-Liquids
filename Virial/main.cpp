#include <iostream>
#include "graph.hpp"

int main() {
    // 2D Graph Example
    std::cout << "=== 2D Graph Example ===\n";
    // Create a graph with 2 nodes (non-complete)
    NDGraph graph2(2, false);
    // Add an edge between node 0 and node 1
    std::cout << "Initial 2D Graph:\n";
    graph2.printGraph();
    graph2.addEdge(0, 1);
    std::cout << "Adding edge now\n";
    graph2.printGraph();

    std::cout << "\nAdding a new node and connecting it in 2D Graph.\n";
    // Add a new node (node id 2)
    int newNode2 = graph2.addNode();
    // Connect node 1 to the new node
    graph2.addEdge(1, newNode2);
    graph2.printGraph();

    std::cout << "\nRemoving edge between node 0 and node 1 in 2D Graph.\n";
    graph2.removeEdge(0, 1);
    graph2.printGraph();

    // 3D Graph Example
    std::cout << "\n=== 3D Graph Example ===\n";
    // Create a graph with 3 nodes (non-complete)
    NDGraph graph3(3, false);
    // Add edges to form a chain: 0->1 and 1->2
    graph3.addEdge(0, 1);
    graph3.addEdge(1, 2);
    std::cout << "Initial 3D Graph:\n";
    graph3.printGraph();

    std::cout << "\nRemoving node 1 from 3D Graph.\n";
    graph3.removeNode(1);
    graph3.printGraph();

    // 4D Graph Example (Complete Graph)
    std::cout << "\n=== 4D Graph Example (Complete Graph) ===\n";
    // Create a complete graph with 4 nodes
    NDGraph graph4(4, true);
    std::cout << "Complete 4D Graph:\n";
    graph4.printGraph();

    std::cout << "\nRemoving edge between node 0 and node 2 in 4D Graph.\n";
    graph4.removeEdge(0, 2);
    graph4.printGraph();
    
    return 0;
}