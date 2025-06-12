#include "GraphGeneration.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

void generateAndSaveGraphs(int maxOrder, const std::string &filename) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    outfile << maxOrder << "\n";
    for (int n = 2; n <= maxOrder; n++) {
        std::vector<NDGraph> graphs = GraphUtils::generateBiconnectedGraphsNoIsomorphism(n);
        outfile << n << " " << graphs.size() << "\n";
        for (auto &graph : graphs) {
            std::vector<Edge> edges = graph.getEdges();
            int numNodes = n; // assuming order equals number of nodes
            int numEdges = edges.size();
            double degeneracy = GraphUtils::computeDegeneracy(graph);
            outfile << numNodes << " " << numEdges << " " << degeneracy << "\n";
            for (auto &edge : edges) {
                outfile << edge.from << " " << edge.to << "\n";
            }
        }
    }
    outfile.close();
    std::cout << "Graphs saved to " << filename << std::endl;
}
