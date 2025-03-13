#ifndef GRAPHTOINTEGRAL_HPP
#define GRAPHTOINTEGRAL_HPP

#include "graph.hpp"
#include "integration.hpp"
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <array>

// Define a function type for a 3D integrand that operates on a vector of 3D points.
using Integrand3D = std::function<double(const std::vector<std::array<double, 3>>&)>;

// Converts an NDGraph to an integrand using the 3D distances between nodes.
inline Integrand3D graphToIntegrand(const NDGraph &graph,
                                    double epsilon, double sigma,
                                    double kb, double T)
{
    const auto &edges = graph.getEdges();

    return [=](const std::vector<std::array<double,3>> &coords) -> double
    {
        double product = 1.0;
        for (const auto &edge : edges)
        {
            double dx = coords[edge.from][0] - coords[edge.to][0];
            double dy = coords[edge.from][1] - coords[edge.to][1];
            double dz = coords[edge.from][2] - coords[edge.to][2];
            double r  = std::sqrt(dx*dx + dy*dy + dz*dz);

            double f = computeMayerFunction(r, epsilon, sigma, kb, T);
            product *= f;
            if (std::fabs(product) < 1e-15) break; // Early exit for negligible products
        }
        return product;
    };
}

/**
 * Computes the cluster integral for a graph via Monte Carlo integration.
 *
 * Integration is performed over a 3*(numNodes-1)-dimensional hypercube,
 * with each free coordinate integrated over the interval [-R,R]. Node 0 is fixed at (0,0,0).
 *
 * This implementation uses the generic monteCarloIntegration function from integration.hpp.
 */
inline double computeGraphIntegral(const NDGraph &graph,
                                   double R,
                                   double epsilon, double sigma,
                                   double kb, double T,
                                   int numSamples = 1000000)
{
    int numNodes = graph.getNumNodes();
    if (numNodes < 1) {
        throw std::runtime_error("Graph must have at least one node.");
    }
    if (numNodes == 1) {
        return 1.0; 
    }

    // For (numNodes-1) free nodes in 3D, total integration dimension is 3*(numNodes-1)
    int freeNodes = numNodes - 1;
    int totalDim = freeNodes * 3;
    // Create integration limits: each coordinate in [-R, R]
    std::vector<std::pair<double, double>> limits(totalDim, { -R, R });

    // Get the 3D integrand from the graph
    auto integrand3D = graphToIntegrand(graph, epsilon, sigma, kb, T);

    // Wrap the 3D integrand to match the generic integrand interface which accepts vector<double>
    auto integrandWrapper = [=](const std::vector<double>& x) -> double {
         // Reconstruct the vector of 3D coordinates.
         std::vector<std::array<double, 3>> coords(numNodes);
         // Fix node 0 at the origin.
         coords[0] = {0.0, 0.0, 0.0};
         // For each free node, pack three consecutive values from x.
         for (int i = 1; i < numNodes; i++) {
             int idx = (i - 1) * 3;
             coords[i] = { x[idx], x[idx+1], x[idx+2] };
         }
         return integrand3D(coords);
    };

    // Use the generic Monte Carlo integration routine.
    return monteCarloIntegration(integrandWrapper, limits, numSamples);
}

#endif // GRAPHTOINTEGRAL_HPP
