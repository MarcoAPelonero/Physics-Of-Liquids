#include "graphUtils.hpp"
#include "graphToIntegrand.hpp"
#include "mcHitOrMiss.hpp"
#include "potentials.hpp"

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

    // Check that the loop graph actually yealds the same integral

    NDGraph loopGraph(4, false);
    loopGraph.addEdge(0, 1);
    loopGraph.addEdge(1, 2);
    loopGraph.addEdge(2, 3);
    loopGraph.addEdge(3, 0);
    loopGraph.addEdge(0, 2);

    NDGraph loopGraph2(4, false);
    loopGraph2.addEdge(0, 1);
    loopGraph2.addEdge(0, 2);
    loopGraph2.addEdge(0, 3);
    loopGraph2.addEdge(1, 2);
    loopGraph2.addEdge(1, 3);

    PotentialFunction potHS = [](double r, double sigma, double epsilon) {
        return HS(r, sigma, epsilon);
    };

    double sigma = 1.0;
    double epsilon = 1.0;
    int dimension = 3;

    auto integrand = graphToIntegrand(loopGraph, potHS, sigma, epsilon, dimension);
    auto integrand2 = graphToIntegrand(loopGraph2, potHS, sigma, epsilon, dimension);

    long nSamples = 1000000;

    double estimate = monteCarloHitOrMiss(
        integrand,
        dimension,
        loopGraph.getNumNodes() - 1,
        sigma,
        nSamples
    );

    double estimate2 = monteCarloHitOrMiss(
        integrand2,
        dimension,
        loopGraph2.getNumNodes() - 1,
        sigma,
        nSamples
    );

    double deg = GraphUtils::computeDegeneracy(loopGraph);
    double deg2 = GraphUtils::computeDegeneracy(loopGraph2);

    double graphValue = estimate * deg;
    double graphValue2 = estimate2 * deg2;

    std::cout << "Loop graph degeneracy: " << deg << std::endl;
    std::cout << "Loop graph 2 degeneracy: " << deg2 << std::endl;

    std::cout << "Loop graph integral * degeneracy: " << graphValue << std::endl;
    std::cout << "Loop graph 2 integral * degeneracy: " << graphValue2 << std::endl;
}