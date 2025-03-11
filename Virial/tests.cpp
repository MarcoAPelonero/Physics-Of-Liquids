#include <iostream>
#include "graph.hpp"
#include "graphUtils.hpp"

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
    
    // ==================================================
    // Additional Examples to Test Graph Properties
    // ==================================================
    std::cout << "\n=== Graph Property Tests ===\n";

    // Test A: Linear chain graph (0-1-2)
    // Expected: isConnected -> true,
    //           isBiconnected -> false (node 1 is an articulation point)
    NDGraph chainGraph(3, false);
    chainGraph.addEdge(0, 1);
    chainGraph.addEdge(1, 2);
    std::cout << "\nChain Graph (0-1-2):\n";
    chainGraph.printGraph();
    std::cout << "isConnected: " << (chainGraph.isConnected() ? "true" : "false") << "\n";
    std::cout << "isBiconnected: "  << (chainGraph.isBiconnected()  ? "true" : "false") << "\n";

    // Test B: Complete Triangle graph (0,1,2 with all edges)
    // Expected: isConnected -> true,
    //           isBiconnected -> true (no articulation points)
    NDGraph triangleGraph(3, false);
    triangleGraph.addEdge(0, 1);
    triangleGraph.addEdge(1, 2);
    triangleGraph.addEdge(0, 2);
    std::cout << "\nComplete Triangle Graph:\n";
    triangleGraph.printGraph();
    std::cout << "isConnected: " << (triangleGraph.isConnected() ? "true" : "false") << "\n";
    std::cout << "isBiconnected: "  << (triangleGraph.isBiconnected()  ? "true" : "false") << "\n";

    // Test C: Disconnected Graph: 3 nodes, no edges
    // Expected: isConnected -> false,
    //           isBiconnected -> false (graph not connected)
    NDGraph disconnectedGraph(3, false);
    std::cout << "\nDisconnected Graph (3 isolated nodes):\n";
    disconnectedGraph.printGraph();
    std::cout << "isConnected: " << (disconnectedGraph.isConnected() ? "true" : "false") << "\n";
    std::cout << "isBiconnected: "  << (disconnectedGraph.isBiconnected()  ? "true" : "false") << "\n";

    // Test D: Square graph (0-1-2-3 with edges 0-1, 1-2, 2-3, 3-0)
    // Expected: isConnected -> true,
    //           isBiconnected -> true (no articulation points)
    NDGraph squareGraph(4, false);
    squareGraph.addEdge(0, 1);
    squareGraph.addEdge(1, 2);
    squareGraph.addEdge(2, 3);
    squareGraph.addEdge(3, 0);
    std::cout << "\nSquare Graph (0-1-2-3):\n";
    squareGraph.printGraph();
    std::cout << "isConnected: " << (squareGraph.isConnected() ? "true" : "false") << "\n";
    std::cout << "isBiconnected: "  << (squareGraph.isBiconnected()  ? "true" : "false") << "\n";

    // ==================================================
    // Biconnected Graphs Generation Tests via GraphUtils
    // ==================================================
    std::cout << "\n=== Biconnected Graphs Generation Tests ===\n";

    // Test for 2-node biconnected graphs
    std::cout << "\nGenerating 2-node biconnected graphs:\n";
    std::vector<NDGraph> biconnected2 = GraphUtils::generateAllBiconnectedGraphsOptimized(2);
    std::cout << "Total 2-node biconnected graphs: " << biconnected2.size() << "\n";
    for (const auto &graph : biconnected2) {
        graph.printGraph();
        std::cout << "isBiconnected: " << (graph.isBiconnected() ? "true" : "false") << "\n";
    }

    // Test for 3-node biconnected graphs
    std::cout << "\nGenerating 3-node biconnected graphs:\n";
    std::vector<NDGraph> biconnected3 = GraphUtils::generateAllBiconnectedGraphsOptimized(3);
    std::cout << "Total 3-node biconnected graphs: " << biconnected3.size() << "\n";
    for (const auto &graph : biconnected3) {
        graph.printGraph();
        std::cout << "isBiconnected: " << (graph.isBiconnected() ? "true" : "false") << "\n";
    }

    // Test for 4-node biconnected graphs
    std::cout << "\nGenerating 4-node biconnected graphs:\n";
    std::vector<NDGraph> biconnected4 = GraphUtils::generateAllBiconnectedGraphsOptimized(4);
    std::cout << "Total 4-node biconnected graphs: " << biconnected4.size() << "\n";
    for (const auto &graph : biconnected4) {
        graph.printGraph();
        std::cout << "isBiconnected: " << (graph.isBiconnected() ? "true" : "false") << "\n";
    }

    return 0;
}