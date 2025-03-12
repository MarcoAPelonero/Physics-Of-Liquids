#include "graph.hpp"
#include "graphUtils.hpp"
#include "integration.hpp"
#include "graphToIntegral.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <max_order> <T> <volume> <range> <numSamples>\n";
        return 1;
    }

    int maxOrder = std::atoi(argv[1]);
    double T     = std::atof(argv[2]);
    double vol   = std::atof(argv[3]);
    double R     = std::atof(argv[4]);
    int numSamples = std::atoi(argv[5]);

    double epsilon = 1.0;
    double sigma   = 1.0;
    double kb      = 1.0;

    for (int n = 1; n <= maxOrder; n++) {
        auto graphs = GraphUtils::generateAllConnectedGraphs(n);
        for (auto &g : graphs) {
            double integralVal = computeGraphIntegral(g, -R, R, epsilon, sigma, kb, T, numSamples);
            std::cout << "Order=" << n << ", Integral=" << integralVal << std::endl;
        }
    }
    return 0;
}