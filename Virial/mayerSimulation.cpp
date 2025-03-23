#include "graph.hpp"
#include "graphUtils.hpp"
#include "graphToIntegrand.hpp"
#include "mayerSampling.hpp"
#include "MonteCarlo.hpp"
#include "potentials.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <filesystem>

#define M_PI 3.14159265358979323846

int main(int argc, char **argv) {

    int order = 3;
    int nSamples = 100000;
    int dimension = 3;
    double sigma = 1.0;
    double epsilon = 1.0;
    double T = 1.0;
    std::string outfileName = "results.txt";

    if (argc > 1) {
        order = std::stoi(argv[1]);
    }
    if (argc > 2) {
        nSamples = std::stoi(argv[2]);
    }
    if (argc > 3) {
        dimension = std::stoi(argv[3]);
    }
    if (argc > 4) {
        sigma = std::stod(argv[4]);
    }
    if (argc > 5) {
        epsilon = std::stod(argv[5]);
    }
    if (argc > 6) {
        T = std::stod(argv[6]);
    }
    if (argc > 7) {
        outfileName = argv[7];
    }

    std::cout << "Order: " << order << std::endl;
    std::cout << "nSamples: " << nSamples << std::endl;
    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "T: " << T << std::endl;

    double beta = 1.0 / T;

    // Make sure the vector has order+1 elements (indexing from 0 to order)
    std::vector<double> virialCoefficientsMayerMetropolis(order+1, 0.0);

    HardSpheresCoefficients hsCoeffs;
    hsCoeffs.changeForm(sigma);
    // Define the reference values for the HS potential

    PotentialFunction potential = [](double r, double sigma, double epsilon) {
      return LJ(r, sigma, epsilon);
    };
    PotentialFunction referencePotential = [](double r, double sigma, double epsilon) {
        return HS(r, sigma, epsilon);
    };

    // Compute all biconnected graphs for each order
    for (int n = 0; n <= order; ++n) {
        if (n < 2) {
            virialCoefficientsMayerMetropolis[n] = 0.0;
            continue;
        }
        
        std::vector<NDGraph> graphs = GraphUtils::generateBiconnectedGraphsNoIsomorphism(n);
        
        double integral = 0.0;

        for (auto &g : graphs) {
            auto [integrandFull, integrandRef, config] = createIntegrandsAndConfig(
                g,
                potential,
                referencePotential,
                sigma,
                epsilon,
                dimension,
                beta
            );
            
            double estimate = MonteCarloMayerMetropolis(
                config,
                integrandFull,
                integrandRef,
                dimension,
                n-1,
                sigma,
                nSamples
            );

            double deg = GraphUtils::computeDegeneracy(g);
            integral += estimate * deg;
        }

        double factor = -(n-1) / ( std::tgamma(n+1) );
        double finalContribution = factor * integral;
        virialCoefficientsMayerMetropolis[n] = finalContribution * hsCoeffs[n];
    }

    double V = (2 * M_PI)/3 * sigma * sigma * sigma;

    std::cout << "Virial coefficients (Mayer Metropolis):" << std::endl;
    for (int i = 0; i <= order; ++i) {
        if (i < 2) {
            std::cout << "0 ";
            continue;
        }
        std::cout << virialCoefficientsMayerMetropolis[i]/pow(V,i-1) << " ";
    }
    std::cout << std::endl;

    // Prepare the output file name.
    // If using the default name, modify it so that it starts with "results_T_"
    if (outfileName == "results.txt") {
        outfileName = "output/results_T_" + std::to_string(T) + ".txt";
    }

    // Ensure that the output directory exists
    std::filesystem::path outputDir = std::filesystem::path(outfileName).parent_path();
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directories(outputDir);
    }
    
    std::ofstream outfile(outfileName);
    if (!outfile) {
        std::cerr << "Error opening file " << outfileName << " for writing." << std::endl;
        return 1;
    }

    // Write the temperature line that the plotter expects
    outfile << "T* = " << T << std::endl;
    
    // Write each virial coefficient line starting from order n=2
    for (int i = 2; i <= order; ++i) {
        outfile << "n=" << i << ": " << virialCoefficientsMayerMetropolis[i]/pow(V,i-1) << std::endl;
    }

    outfile.close();
    std::cout << "Results written to " << outfileName << std::endl;

    return 0;
}
