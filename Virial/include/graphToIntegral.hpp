#ifndef GRAPHTOINTEGRAL_HPP
#define GRAPHTOINTEGRAL_HPP

#include "graph.hpp"
#include "integration.hpp"
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>

// Converts an NDGraph into an integrand function.
// Each node corresponds to one integration coordinate (assumed one-dimensional).
// Each edge in the graph contributes a Mayer function computed from the absolute difference 
// between the coordinates of the two nodes connected by that edge. The overall integrand is the 
// product of the Mayer functions for all edges.
inline Integrand graphToIntegrand(const NDGraph &graph,
                                  double epsilon, double sigma,
                                  double kb, double T) {
    const auto &edges = graph.getEdges();
    int numNodes = graph.getNumNodes();

    return [=](const std::vector<double> &coords) -> double {
        if (coords.size() != static_cast<size_t>(numNodes)) {
            throw std::runtime_error("Number of integration coordinates must equal number of graph nodes.");
        }
        double product = 1.0;
        for (const auto &edge : edges) {
            double distance = std::abs(coords[edge.from] - coords[edge.to]);
            double f = computeMayerFunction(distance, epsilon, sigma, kb, T);
            product *= f;
        }
        return product;
    };
}

// Computes the integral corresponding to the given NDGraph over a hypercube domain.
// Instead of a matrix of limits, we assume each node is integrated over the same interval [a,b].
// The integration is performed using Monte Carlo integration.
inline double computeGraphIntegral(const NDGraph &graph,
                                   double a, double b,
                                   double epsilon, double sigma,
                                   double kb, double T,
                                   int numSamples = 1000000) {
    int numNodes = graph.getNumNodes();
    // Generate hypercube limits: same [a, b] for each node.
    std::vector<std::pair<double, double>> limits(numNodes, std::make_pair(a, b));
    
    Integrand integrand = graphToIntegrand(graph, epsilon, sigma, kb, T);
    return monteCarloIntegration(integrand, limits, numSamples);
}

#endif // GRAPHTOINTEGRAL_HPP
