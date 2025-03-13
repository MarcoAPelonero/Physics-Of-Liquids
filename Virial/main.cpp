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

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <max_order> <T> <range_R> <numSamples>\n";
        return 1;
    }

    int    maxOrder   = std::atoi(argv[1]);
    double T          = std::atof(argv[2]);
    double R          = std::atof(argv[3]);  // half-side for [-R,R] in 3D
    int    numSamples = std::atoi(argv[4]);

    // Lennard-Jones parameters in reduced units
    double epsilon = 1.0;
    double sigma   = 1.0;
    double kb      = 1.0;

    std::ofstream outFile("virial_coefficients.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: could not open output file\n";
        return 1;
    }

    // Compute from n=2 up to maxOrder (n=1 is trivial: integral=1)
    for (int n = 2; n <= maxOrder; n++) {
        // Generate the graphs (assuming generateAllBiconnectedGraphsOptimized is defined properly)
        auto graphs = GraphUtils::generateAllBiconnectedGraphsOptimized(n, false);

        double Bn = 0.0;
        for (auto &graph : graphs) {
            // Use the integration library via computeGraphIntegral
            double graphIntegral = computeGraphIntegral(
                graph,     // The graph
                R,         // half-width for 3D sampling (each free node is in [-R,R]^3)
                epsilon,
                sigma,
                kb,
                T,
                numSamples // Number of Monte Carlo samples
            );

            int symmetryFactor = graph.getDegeneracy(); // Automorphism-based degeneracy factor
            // Standard factor: (-1)^(n-1)/(n * symmetryFactor)
            double weight = std::pow(-1.0, n-1) / (static_cast<double>(n) * symmetryFactor);

            Bn += weight * graphIntegral;
        }

        // Output the virial coefficient for order n.
        outFile << n << " " << Bn << "\n";
        std::cout << "Computed virial coefficient for n = " << n 
                  << ": " << Bn << "\n";
    }

    outFile.close();
    return 0;
}
