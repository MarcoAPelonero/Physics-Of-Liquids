#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "MonteCarlo.hpp"

#define M_PI 3.14159265358979323846

// Simple Edge structure to hold graph edge data.
struct Edge {
    int from;
    int to;
};

// Structure to hold one graph (as loaded from file).
struct GraphData {
    int numNodes;
    int numEdges;
    double degeneracy;
    std::vector<Edge> edges;
};

// Structure to hold all graphs for one order.
struct OrderGraphs {
    int order;
    std::vector<GraphData> graphs;
};

// Loads graphs from a file (format as written by GraphGeneration.cpp).
std::vector<OrderGraphs> loadGraphs(const std::string &filename) {
    std::vector<OrderGraphs> orders;
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return orders;
    }
    int maxOrder;
    infile >> maxOrder;
    for (int n = 2; n <= maxOrder; n++) {
        OrderGraphs orderGraph;
        int order, numGraphs;
        infile >> order >> numGraphs;
        orderGraph.order = order;
        for (int i = 0; i < numGraphs; i++) {
            GraphData g;
            infile >> g.numNodes >> g.numEdges >> g.degeneracy;
            for (int j = 0; j < g.numEdges; j++) {
                Edge e;
                infile >> e.from >> e.to;
                g.edges.push_back(e);
            }
            orderGraph.graphs.push_back(g);
        }
        orders.push_back(orderGraph);
    }
    infile.close();
    return orders;
}

int main(int argc, char** argv) {
    std::string graphFile = "graphs.dat";
    int dimension = 3;
    double sigma = 1.0;
    double epsilon = 1.0;
    double T = 1.0;
    long nSamples = 1000000;

    if (argc > 1) graphFile = argv[1];
    if (argc > 2) nSamples = std::atol(argv[2]);
    if (argc > 3) dimension = std::atoi(argv[3]);
    if (argc > 4) sigma = std::atof(argv[4]);
    if (argc > 5) epsilon = std::atof(argv[5]);
    if (argc > 6) T = std::atof(argv[6]);

    double beta = 1.0 / T;
    double sideLength = 5.0 * sigma;

    std::vector<OrderGraphs> orders = loadGraphs(graphFile);
    if (orders.empty()) {
        std::cerr << "No graphs loaded." << std::endl;
        return 1;
    }

    // Prepare a vector for virial coefficients; indices 0,1 unused.
    std::vector<double> virialCoefficients(orders.size() + 2, 0.0);

    for (auto &orderGraph : orders) {
        int n = orderGraph.order;
        double integralOrder = 0.0;
        for (auto &graph : orderGraph.graphs) {
            // Prepare host-side edge arrays.
            std::vector<int> h_edgeFrom;
            std::vector<int> h_edgeTo;
            for (auto &edge : graph.edges) {
                h_edgeFrom.push_back(edge.from);
                h_edgeTo.push_back(edge.to);
            }
            int nFreeNodes = n - 1;  // with node 0 fixed.
            double integralEstimate = runMonteCarloIntegration(dimension,
                                                               nFreeNodes,
                                                               sigma,
                                                               epsilon,
                                                               beta,
                                                               nSamples,
                                                               sideLength,
                                                               h_edgeFrom.data(),
                                                               h_edgeTo.data(),
                                                               graph.numEdges);
            integralOrder += graph.degeneracy * integralEstimate;
        }
        // Apply the overall prefactor: -(n-1)/gamma(n+1)
        double factor = -(n - 1) / tgamma(n + 1);
        double finalContribution = factor * integralOrder;
        virialCoefficients[n] = finalContribution;
        std::cout << "Order n = " << n << " virial coefficient: " << finalContribution << std::endl;
    }

    // Convert virial coefficients to a form in terms of packing fraction.
    double coeff = (M_PI / 3.0) * 2;
    double v0 = coeff * sigma * sigma * sigma;
    for (size_t i = 2; i < virialCoefficients.size(); i++) {
        double term = pow(v0, i - 1);
        double virialCoeffPF = virialCoefficients[i] / term;
        std::cout << "n = " << i << " virial coefficient (packing fraction): " << virialCoeffPF << std::endl;
    }

    return 0;
}
