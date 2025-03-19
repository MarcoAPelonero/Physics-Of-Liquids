#include "graphUtils.hpp"
#include <vector>
#include <iostream>

int main() {

    std::cout << "Testing generation of biconnected graphs for n = 4\n";
    std::vector<NDGraph> graphs1 = GraphUtils::generateBiconnectedGraphsNoIsomorphism(4);

    for (auto &g : graphs1) {
        
        g.printGraph();
        int deg = GraphUtils::computeDegeneracy(g);
        std::cout << "Degeneracy: " << deg << std::endl;
    }
}