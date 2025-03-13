// test_graphutils_degeneracy.cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include "graph.hpp"
#include "graphUtils.hpp"

//-------------------------------------------------------------
// Test for biconnected graphs degeneracy for n = 2, 3, 4.
//-------------------------------------------------------------

// For n = 2: K2 should be the only biconnected graph with 1 edge and degeneracy 1.
void testBiconnectedDegeneracy_n2() {
    std::cout << "=== Testing biconnected graphs degeneracy for n = 2 ===\n";
    auto graphs = GraphUtils::generateAllBiconnectedGraphsOptimized(2, false);
    // Expect exactly one biconnected graph (K2)
    assert(graphs.size() == 1 && "For n=2, expected exactly 1 biconnected graph.");
    
    NDGraph g = graphs[0];
    std::cout << "Graph (n=2):\n";
    g.printGraph();
    // K2 has 1 edge and its degeneracy must be 1.
    std::cout << "Number of edges: " << g.getNumEdges() 
              << " | Expected degeneracy: 1 | Computed degeneracy: " << g.getDegeneracy() << "\n";
    assert(g.getNumEdges() == 1);
    assert(g.getDegeneracy() == 1);
}

// For n = 3: K3 should be the only biconnected graph with 3 edges and degeneracy 2.
void testBiconnectedDegeneracy_n3() {
    std::cout << "=== Testing biconnected graphs degeneracy for n = 3 ===\n";
    auto graphs = GraphUtils::generateAllBiconnectedGraphsOptimized(3, false);
    // Expect exactly one biconnected graph (K3)
    assert(graphs.size() == 1 && "For n=3, expected exactly 1 biconnected graph.");
    
    NDGraph g = graphs[0];
    std::cout << "Graph (n=3):\n";
    g.printGraph();
    // K3 has 3 edges and its degeneracy must be 2.
    std::cout << "Number of edges: " << g.getNumEdges() 
              << " | Expected degeneracy: 2 | Computed degeneracy: " << g.getDegeneracy() << "\n";
    assert(g.getNumEdges() == 3);
    assert(g.getDegeneracy() == 2);
}

// For n = 4: We expect three non-isomorphic biconnected graphs.
//   - One graph with 4 edges (4-cycle, C4)  → expected degeneracy: 2
//   - One graph with 5 edges (diamond, K4 minus one edge) → expected degeneracy: 2
//   - One graph with 6 edges (complete graph, K4) → expected degeneracy: 3
void testBiconnectedDegeneracy_n4() {
    std::cout << "=== Testing biconnected graphs degeneracy for n = 4 ===\n";
    auto graphs = GraphUtils::generateAllBiconnectedGraphsOptimized(4, false);
    std::cout << "Total biconnected graphs for n=4: " << graphs.size() << "\n";
    assert(graphs.size() == 3 && "For n=4, expected exactly 3 non-isomorphic biconnected graphs.");
    
    // Iterate and check each graph
    for (auto &g : graphs) {
        int numEdges = g.getNumEdges();
        int expectedDegeneracy = (numEdges == 6) ? 3 : 2;
        std::cout << "Graph details:\n";
        g.printGraph();
        std::cout << "Number of edges: " << numEdges 
                  << " | Expected degeneracy: " << expectedDegeneracy 
                  << " | Computed degeneracy: " << g.getDegeneracy() << "\n\n";
        assert((numEdges == 4 || numEdges == 5 || numEdges == 6) && "Unexpected edge count for n=4 graph.");
        assert(g.getDegeneracy() == expectedDegeneracy);
    }
}

int main() {
    std::cout << "Starting GraphUtils degeneracy tests...\n";
    testBiconnectedDegeneracy_n2();
    testBiconnectedDegeneracy_n3();
    testBiconnectedDegeneracy_n4();
    std::cout << "All GraphUtils degeneracy tests passed successfully.\n";
    return 0;
}
