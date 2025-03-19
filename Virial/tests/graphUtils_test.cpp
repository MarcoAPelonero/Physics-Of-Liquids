#include "graph.hpp"
#include "graphUtils.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>

int main() {
    // Test the generation of biconnected (non-isomorphic) graphs for n = 4
    std::cout << "Testing generation of biconnected graphs for n = 4\n";
    auto begin = std::chrono::steady_clock::now();
    std::vector<NDGraph> graphs1 = GraphUtils::generateBiconnectedGraphsNoIsomorphism(4);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time difference for brute force method= "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << "[ms]" << std::endl;

    // Print the generated graphs
    for (auto &g : graphs1) {
        g.printGraph();
    }

    // Check if the number of generated graphs for n = 4 is correct
    if (graphs1.size() != 3) {
        std::cerr << "Error: Incorrect number of graphs generated for n = 4\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    // Baseline tests for n = 2, 3, 4
    // ------------------------------------------------------------------------
    std::cout << "Baseline tests\n";

    // For n = 2, the complete graph K2 is the only 2-node biconnected graph
    NDGraph g2(2, true); // K2

    // For n = 3, the complete graph K3 is the only 3-node biconnected graph
    NDGraph g3(3, true); // K3

    // For n = 4, we consider:
    // 1) g4_1: K4 (complete graph)
    // 2) g4_2: K4 minus one edge (remove (0,2))
    // 3) g4_3: K4 minus two edges (remove (0,2) and (1,3)) => forms a 4-cycle
    NDGraph g4_1(4, true); // K4
    NDGraph g4_2(4, true); // K4
    NDGraph g4_3(4, true); // K4

    g4_2.removeEdge(0, 2);
    g4_3.removeEdge(0, 2);
    g4_3.removeEdge(1, 3);

    // Generate non-isomorphic biconnected graphs for n=2, 3, 4
    std::vector<NDGraph> generated2 = GraphUtils::generateBiconnectedGraphsNoIsomorphism(2);
    std::vector<NDGraph> generated3 = GraphUtils::generateBiconnectedGraphsNoIsomorphism(3);
    std::vector<NDGraph> generated4 = GraphUtils::generateBiconnectedGraphsNoIsomorphism(4);

    // ------------------------------------------------------------------------
    // Check for n = 2
    // ------------------------------------------------------------------------
    if (generated2.size() != 1) {
        std::cerr << "Error: Incorrect number of graphs generated for n = 2\n";
        return 1;
    }
    if (!(generated2[0] == g2)) {
        std::cerr << "Error: Incorrect graph generated for n = 2\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    // Check for n = 3
    // ------------------------------------------------------------------------
    if (generated3.size() != 1) {
        std::cerr << "Error: Incorrect number of graphs generated for n = 3\n";
        return 1;
    }
    if (!(generated3[0] == g3)) {
        std::cerr << "Error: Incorrect graph generated for n = 3\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    // Check for n = 4
    // ------------------------------------------------------------------------
    if (generated4.size() != 3) {
        std::cerr << "Error: Incorrect number of graphs generated for n = 4\n";
        return 1;
    }

    // Visual check
    std::cout << "Visual checks for generated graphs for n = 4\n";
    std::cout << "Generated graphs:\n";
    for (auto &g : generated4) {
        g.printGraph();
    }

    std::cout << "Baseline graphs:\n";
    g4_1.printGraph();
    g4_2.printGraph();
    g4_3.printGraph();

    // ------------------------------------------------------------------------
    // Degeneracy checks
    // For n=2: K2 => degeneracy should be 1
    // For n=3: K3 => degeneracy should be 1
    // For the 3 distinct n=4 graphs:
    //    K4 => degeneracy = 1
    //    K4 minus 1 edge => degeneracy = 6
    //    4-cycle (K4 minus 2 edges) => degeneracy = 3
    // ------------------------------------------------------------------------
    int deg2 = GraphUtils::computeDegeneracy(g2);

    if (deg2 != 1) {
        std::cerr << "Error: Incorrect degeneracy for n = 2\n";
        std::cout << "Computed degeneracy: " << deg2 << std::endl;
        return 1;
    }

    int deg3 = GraphUtils::computeDegeneracy(g3);

    if (deg3 != 1) {
        std::cerr << "Error: Incorrect degeneracy for n = 3\n";
        return 1;
    }

    int deg4_1 = GraphUtils::computeDegeneracy(g4_1);
    int deg4_2 = GraphUtils::computeDegeneracy(g4_2);
    int deg4_3 = GraphUtils::computeDegeneracy(g4_3);

    if (deg4_1 != 1) {
        std::cerr << "Error: Incorrect degeneracy for n = 4, graph 1 (K4)\n";
        return 1;
    }
    if (deg4_2 != 6) {
        std::cerr << "Error: Incorrect degeneracy for n = 4, graph 2 (K4 minus an edge)\n";
        return 1;
    }
    if (deg4_3 != 3) {
        std::cerr << "Error: Incorrect degeneracy for n = 4, graph 3 (4-cycle)\n";
        return 1;
    }

    
    // All tests passed
    std::cout << "All tests passed successfully!\n";
    return 0;
}
