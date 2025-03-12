#include <iostream>
#include "graph.hpp"
#include "graphUtils.hpp"
#include "integration.hpp"
#include "graphToIntegral.hpp"
#include <chrono>

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

    // ==========================================
    // Compare Optimized vs. Non-Optimized Times
    // ==========================================
    auto compareBiconnectedTimes = [&](int n) {
        std::cout << "\n=== Biconnected Graphs Time Comparison for n = " << n << " ===\n";

        // Measure non-optimized
        auto startNonOpt = std::chrono::high_resolution_clock::now();
        std::vector<NDGraph> nonOptResults = GraphUtils::generateAllBiconnectedGraphs(n);
        auto endNonOpt = std::chrono::high_resolution_clock::now();
        auto nonOptMs = std::chrono::duration_cast<std::chrono::milliseconds>(endNonOpt - startNonOpt).count();

        // Measure optimized
        auto startOpt = std::chrono::high_resolution_clock::now();
        std::vector<NDGraph> optResults = GraphUtils::generateAllBiconnectedGraphsOptimized(n);
        auto endOpt = std::chrono::high_resolution_clock::now();
        auto optMs = std::chrono::duration_cast<std::chrono::milliseconds>(endOpt - startOpt).count();

        // Print results
        std::cout << "Non-Optimized found " << nonOptResults.size() << " graphs in " << nonOptMs << " ms\n";
        std::cout << "Optimized found " << optResults.size() << " graphs in " << optMs   << " ms\n";
        std::cout << "Difference: " << (nonOptMs - optMs) << " ms\n";
    };

    // Compare for n = 2, 3, 4
    compareBiconnectedTimes(2);
    compareBiconnectedTimes(3);
    compareBiconnectedTimes(4);
    compareBiconnectedTimes(5);
    // compareBiconnectedTimes(6);
    // Parameters for the Lennard-Jones potential and Mayer function
    double epsilon = 1.0; // Depth of the potential well
    double sigma   = 1.0; // Finite distance at which the potential is zero
    double kb      = 1.0; // Boltzmann constant (using reduced units)
    double T       = 1.0; // Temperature

    // Define integration limits for the free particles.
    // Here we assume one-dimensional positions for the free particles.
    // We fix particle 1 at x = 0, and allow the others to vary over [-5, 5].
    const double L = 15.0;

    // ============================================================
    // Example 1: Three-particle cluster
    // Fix particle 1 at x = 0; integrate over x2 and x3.
    // Mayer functions:
    // f12: between particle 1 (0) and particle 2 (x2)
    // f13: between particle 1 (0) and particle 3 (x3)
    // f23: between particle 2 (x2) and particle 3 (x3)
    // ============================================================
    Integrand threeParticleIntegrand = [=](const std::vector<double>& x) -> double {
        // x[0] = x2, x[1] = x3
        double x2 = x[0];
        double x3 = x[1];

        // Distances (absolute value since we are in one dimension)
        double r12 = std::abs(x2 - 0.0);  // Particle 1 at 0
        double r13 = std::abs(x3 - 0.0);
        double r23 = std::abs(x3 - x2);

        double f12 = computeMayerFunction(r12, epsilon, sigma, kb, T);
        double f13 = computeMayerFunction(r13, epsilon, sigma, kb, T);
        double f23 = computeMayerFunction(r23, epsilon, sigma, kb, T);

        return f12 * f13 * f23;
    };

    // Two integration variables: x2 and x3 over [-L, L]
    std::vector<std::pair<double, double>> limitsThree = { {-L, L}, {-L, L} };

    int samplesThree = 500000; // adjust sample count as needed
    double resultThree = monteCarloIntegration(threeParticleIntegrand, limitsThree, samplesThree);
    std::cout << "Monte Carlo integration result (3-particle cluster): " << resultThree << std::endl;

    // ============================================================
    // Example 2: Four-particle cluster
    // Fix particle 1 at x = 0; integrate over x2, x3, and x4.
    // Mayer functions:
    // f12: between particle 1 (0) and particle 2 (x2)
    // f13: between particle 1 (0) and particle 3 (x3)
    // f14: between particle 1 (0) and particle 4 (x4)
    // f23: between particle 2 (x2) and particle 3 (x3)
    // f24: between particle 2 (x2) and particle 4 (x4)
    // f34: between particle 3 (x3) and particle 4 (x4)
    // ============================================================
    Integrand fourParticleIntegrand = [=](const std::vector<double>& x) -> double {
        // x[0] = x2, x[1] = x3, x[2] = x4
        double x2 = x[0];
        double x3 = x[1];
        double x4 = x[2];

        // Compute distances
        double r12 = std::abs(x2 - 0.0);
        double r13 = std::abs(x3 - 0.0);
        double r14 = std::abs(x4 - 0.0);
        double r23 = std::abs(x3 - x2);
        double r24 = std::abs(x4 - x2);
        double r34 = std::abs(x4 - x3);

        double f12 = computeMayerFunction(r12, epsilon, sigma, kb, T);
        double f13 = computeMayerFunction(r13, epsilon, sigma, kb, T);
        double f14 = computeMayerFunction(r14, epsilon, sigma, kb, T);
        double f23 = computeMayerFunction(r23, epsilon, sigma, kb, T);
        double f24 = computeMayerFunction(r24, epsilon, sigma, kb, T);
        double f34 = computeMayerFunction(r34, epsilon, sigma, kb, T);

        return f12 * f13 * f14 * f23 * f24 * f34;
    };

    // Three integration variables: x2, x3, x4 over [-L, L]
    std::vector<std::pair<double, double>> limitsFour = { {-L, L}, {-L, L}, {-L, L} };

    int samplesFour = 500000; // adjust sample count as needed
    double resultFour = monteCarloIntegration(fourParticleIntegrand, limitsFour, samplesFour);
    std::cout << "Monte Carlo integration result (4-particle cluster): " << resultFour << std::endl;

    // Example: Create a graph for a three-particle cluster.
    // We'll fix node 0 and let nodes 1 and 2 be free.
    // The graph will have nodes 0, 1, 2 and edges: (0,1), (0,2), and (1,2).
    NDGraph graph(3, false);
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 2);
    // (Edges can be added as required by your project.)

    // Define the hypercube integration domain: each node is integrated over [-5, 5].
    double lowerBound = -5.0;
    double upperBound = 5.0;

    int numSamples = 10000000; // Number of Monte Carlo samples.
    double result = computeGraphIntegral(graph, lowerBound, upperBound,
                                         epsilon, sigma, kb, T, numSamples);

    std::cout << "Computed graph integral over hypercube: " << result << std::endl;
    return 0;
}